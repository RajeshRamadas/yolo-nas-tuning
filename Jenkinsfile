pipeline {
    agent any
    environment {
        PYTHON_ENV = "yolo_nni_env"
    }
    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/your-repo/yolo-nas-tuning.git'
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
                    def experimentId = nniOutput =~ /experimentId is ([a-zA-Z0-9]+)/
                    if (experimentId) {
                        env.NNI_EXPERIMENT_ID = experimentId[0][1]
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
                    // Get the best model based on NNI experiment results
                    def bestTrialId = sh(script: "source $PYTHON_ENV/bin/activate && nnictl experiment export ${env.NNI_EXPERIMENT_ID} | grep -oP '\"id\": \"\\K[^\"]+' | head -1", returnStdout: true).trim()
                    echo "Best Trial ID: ${bestTrialId}"

                    // Find the corresponding Ultralytics run directory
                    def bestModel = sh(script: "ls -t runs/train | head -n 1", returnStdout: true).trim()
                    echo "Best Model: ${bestModel}"

                    // Ensure the deployment directory exists
                    sh "mkdir -p deployment"

                    // Copy the best model to the deployment directory
                    sh "cp runs/train/${bestModel}/weights/best.pt deployment/"

                    // Verify the file was copied successfully
                    sh "ls -la deployment/best.pt"
                }
            }
        }
    }
    post {
        always {
            // Clean up NNI experiment
            sh "source $PYTHON_ENV/bin/activate && nnictl stop ${env.NNI_EXPERIMENT_ID} || true"
        }
        success {
            echo "Training & Deployment Completed Successfully"
        }
        failure {
            echo "Training Failed!"
        }
    }
}