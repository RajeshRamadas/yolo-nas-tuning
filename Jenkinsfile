pipeline {
    agent any
    environment {
        PYTHON_ENV = "yolo_nni_env"
    }
    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/RajeshRamadas/yolo-nas-tuning.git'
            }
        }
        stage('Setup Environment') {
            steps {
                sh 'python3 -m venv $PYTHON_ENV'
                sh 'source $PYTHON_ENV/bin/activate && pip install -r requirements.txt'
            }
        }
        stage('Run NAS and Hyperparameter Tuning') {
            steps {
                script {
                    // Start NNI experiment and capture the experiment ID
                    def nniOutput = sh(script: 'source $PYTHON_ENV/bin/activate && nnictl create --config config.yml', returnStdout: true).trim()
                    def experimentIdMatcher = nniOutput =~ /experimentId is ([a-zA-Z0-9]+)/
                    if (experimentIdMatcher) {
                        env.NNI_EXPERIMENT_ID = experimentIdMatcher[0][1]
                        echo "Started NNI experiment with ID: ${env.NNI_EXPERIMENT_ID}"
                    } else {
                        error "Failed to extract experiment ID from NNI output"
                    }
                }
            }
        }
        stage('Monitor Training') {
            steps {
                script {
                    // Wait for NNI experiment to complete
                    def isCompleted = false
                    def maxAttempts = 720 // 6 hours at 30-second intervals
                    def attempt = 0

                    while (!isCompleted && attempt < maxAttempts) {
                        def status = sh(script: "source $PYTHON_ENV/bin/activate && nnictl status ${env.NNI_EXPERIMENT_ID} | grep Status", returnStdout: true).trim()
                        echo "Current experiment status: ${status}"

                        if (status.contains("DONE") || status.contains("ERROR") || status.contains("STOPPED")) {
                            isCompleted = true
                            echo "NNI experiment completed with status: ${status}"
                        } else {
                            sleep(30) // Check every 30 seconds
                            attempt++
                        }
                    }

                    if (!isCompleted) {
                        error "NNI experiment did not complete within the allocated time"
                    }
                }
            }
        }
        stage('Deploy Best Model') {
            steps {
                script {
                    // Get the best trial using NNI's export command and properly parse JSON output
                    sh "source $PYTHON_ENV/bin/activate && nnictl experiment export ${env.NNI_EXPERIMENT_ID} -t json > nni_results.json"

                    // Use proper JSON parsing instead of grep
                    def nniResults = readJSON file: 'nni_results.json'
                    def bestTrial = null
                    def bestAccuracy = -1

                    // Find the best trial based on accuracy
                    for (trial in nniResults.trials) {
                        if (trial.accuracy != null && trial.accuracy > bestAccuracy && trial.status == "SUCCEEDED") {
                            bestAccuracy = trial.accuracy
                            bestTrial = trial
                        }
                    }

                    if (bestTrial == null) {
                        error "No successful trials found in NNI experiment"
                    }

                    def bestTrialId = bestTrial.id
                    echo "Best Trial ID: ${bestTrialId}"
                    echo "Best Trial Accuracy (mAP): ${bestAccuracy}"

                    // The parameters used by the best trial
                    echo "Best Trial Parameters: ${bestTrial.parameter}"

                    // Find the corresponding Ultralytics run directory by directly accessing the trial's output file
                    def trialLogDir = "logs/${env.NNI_EXPERIMENT_ID}/trials/${bestTrialId}"
                    def trialOutput = sh(script: "cat ${trialLogDir}/stdout | grep 'Training completed successfully' -A 10", returnStdout: true).trim()

                    // Extract the run path from the logs
                    def runPathMatcher = trialOutput =~ /runs\/train\/([^ ]+)/
                    def bestModelPath = ""

                    if (runPathMatcher) {
                        def runDir = runPathMatcher[0][1]
                        bestModelPath = "runs/train/${runDir}/weights/best.pt"
                        echo "Best Model Path: ${bestModelPath}"

                        // Verify the path exists
                        def modelExists = sh(script: "test -f ${bestModelPath} && echo 'true' || echo 'false'", returnStdout: true).trim()

                        if (modelExists == 'true') {
                            // Ensure the deployment directory exists
                            sh "mkdir -p deployment"

                            // Copy the best model to the deployment directory
                            sh "cp ${bestModelPath} deployment/best.pt"

                            // Verify the file was copied successfully
                            sh "ls -la deployment/best.pt"

                            // Save the parameters used for the best model
                            writeJSON file: 'deployment/best_parameters.json', json: bestTrial.parameter
                        } else {
                            error "Best model file not found at ${bestModelPath}"
                        }
                    } else {
                        // Fallback method if we can't find the path in logs
                        echo "Could not find exact run path in logs. Using latest run directory..."

                        def latestRun = sh(script: "ls -t runs/train | head -n 1", returnStdout: true).trim()
                        bestModelPath = "runs/train/${latestRun}/weights/best.pt"

                        // Verify the path exists
                        def modelExists = sh(script: "test -f ${bestModelPath} && echo 'true' || echo 'false'", returnStdout: true).trim()

                        if (modelExists == 'true') {
                            sh "mkdir -p deployment"
                            sh "cp ${bestModelPath} deployment/best.pt"
                            sh "ls -la deployment/best.pt"
                            writeJSON file: 'deployment/best_parameters.json', json: bestTrial.parameter
                        } else {
                            error "No model file found"
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            // Clean up NNI experiment
            sh "source $PYTHON_ENV/bin/activate && nnictl stop ${env.NNI_EXPERIMENT_ID} || true"

            // Archive artifacts
            archiveArtifacts artifacts: 'deployment/**', fingerprint: true
            archiveArtifacts artifacts: 'nni_results.json', fingerprint: true
        }
        success {
            echo "Training & Deployment Completed Successfully"
        }
        failure {
            echo "Training Failed!"
        }
    }
}