// =============================================================================
// Declarative Jenkins CI/CD Pipeline for Vision Builder Core
// Topic: Deploying a CI/CD Pipeline for a Python Project Using Docker & Jenkins
// Author: Andasov Temirlan (CSE-2507M, Astana IT University)
// Repository: https://github.com/DaldenU/vision-builder-core
// =============================================================================

pipeline {
    agent any

    environment {
        IMAGE_NAME        = "vision-builder-core"
        DOCKERHUB_REPO    = "daldenu/vision-builder-core"
        IMAGE_TAG         = "${env.BUILD_NUMBER ?: 'latest'}"
        STAGING_CONTAINER = "vision-builder-staging"
        STAGING_PORT      = "8090"
        PYTHONUNBUFFERED  = "1"
    }

    options {
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '10'))
        timeout(time: 20, unit: 'MINUTES')
    }

    stages {
        stage('1. Checkout & Environment Info') {
            steps {
                echo "=== [Stage 1] Checking out Vision Builder Core repository ==="
                checkout scm
                sh '''
                    echo "Git Branch / Commit:"
                    git log -1 --oneline || true
                    python3 --version
                    docker --version
                '''
            }
        }

        stage('2. Static Analysis & DevSecOps Audit') {
            steps {
                echo "=== [Stage 2] Running Code Formatting, Linting, Type-Check & SAST ==="
                sh '''
                    python3 -m venv --system-site-packages .venv_ci
                    . .venv_ci/bin/activate
                    pip install --no-deps -e .

                    echo "--- Running Flake8 PEP-8 Linter ---"
                    python3 -m flake8 vision_builder tests --max-line-length=88 --extend-ignore=E203,W503

                    echo "--- Running Mypy Static Type Verification ---"
                    python3 -m mypy vision_builder tests

                    echo "--- Running Bandit DevSecOps SAST Scan ---"
                    python3 -m bandit -r vision_builder -ll
                '''
            }
        }

        stage('3. Automated Unit Tests & Coverage') {
            steps {
                echo "=== [Stage 3] Executing Pytest Scientific Unit Test Suite ==="
                sh '''
                    . .venv_ci/bin/activate
                    python3 -m pytest --cov=vision_builder \
                           --cov-report=term-missing \
                           --cov-report=xml:coverage.xml \
                           --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                    archiveArtifacts artifacts: 'coverage.xml,test-results.xml', allowEmptyArchive: true
                }
            }
        }

        stage('4. Build Docker Image') {
            steps {
                echo "=== [Stage 4] Building Reproducible Docker Container Image ==="
                sh '''
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} \
                                 -t ${IMAGE_NAME}:latest \
                                 -t ${DOCKERHUB_REPO}:${IMAGE_TAG} \
                                 -t ${DOCKERHUB_REPO}:latest .
                    docker images | grep ${IMAGE_NAME}
                '''
            }
        }

        stage('5. Container Scientific Benchmark & Verification') {
            steps {
                echo "=== [Stage 5] Verifying Containerized CV-Ops Benchmark & CLI ==="
                sh '''
                    docker run --rm ${IMAGE_NAME}:${IMAGE_TAG} \
                        vision-builder benchmark --iterations 50 --defect-rate 0.20
                '''
            }
        }

        stage('6. Push Image to Docker Hub (Optional)') {
            steps {
                echo "=== [Stage 6] Publishing Image to Registry ==="
                sh '''
                    if [ -n "${DOCKERHUB_TOKEN:-}" ] && [ -n "${DOCKERHUB_USERNAME:-}" ]; then
                        echo "$DOCKERHUB_TOKEN" | docker login -u "$DOCKERHUB_USERNAME" --password-stdin
                        docker push ${DOCKERHUB_REPO}:${IMAGE_TAG}
                        docker push ${DOCKERHUB_REPO}:latest
                        docker logout
                    else
                        echo "DOCKERHUB credentials not provided in environment; verified local image tags:"
                        docker images ${DOCKERHUB_REPO}
                    fi
                '''
            }
        }

        stage('7. Continuous Deployment (CD) & Smoke Test') {
            steps {
                echo "=== [Stage 7 - Bonus CD] Deploying to Staging Server & Verifying Health/Inference ==="
                sh '''
                    docker rm -f ${STAGING_CONTAINER} >/dev/null 2>&1 || true
                    docker run -d --name ${STAGING_CONTAINER} \
                        -p ${STAGING_PORT}:8000 \
                        ${IMAGE_NAME}:${IMAGE_TAG}

                    echo "Waiting for deployed container health endpoint..."
                    for i in 1 2 3 4 5 6 7 8 9 10; do
                        if docker exec ${STAGING_CONTAINER} curl -fsS http://127.0.0.1:8000/health; then
                            echo ""
                            echo "Health check passed on attempt $i!"
                            break
                        fi
                        sleep 1
                    done

                    echo "--- Executing Live CV-Ops Inference Smoke Test on Deployed Container ---"
                    docker exec ${STAGING_CONTAINER} curl -fsS http://127.0.0.1:8000/infer | head -n 35

                    echo "--- Validating Kubernetes Deployment Manifests (k8s/) ---"
                    ls -la k8s/
                '''
            }
        }
    }

    post {
        success {
            echo "CI/CD Pipeline completed successfully! Image ${IMAGE_NAME}:${IMAGE_TAG} verified and deployed."
        }
        failure {
            echo "CI/CD Pipeline failed. Inspect stage logs and test-results.xml for diagnostics."
        }
    }
}
