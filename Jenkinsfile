pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        script {
          make test
        }

        pysh(script: 'tox', returnStatus: true, returnStdout: true)
      }
    }
  }
}