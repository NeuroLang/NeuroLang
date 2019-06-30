<<<<<<< 569359a413244715c3fd382bf3f73a76dacc8e34
pipeline {
  agent any
  stages {
    stage('step') {
      steps {
        script {
          sh '''
python -c '
import os
import tox

os.chdir(os.getenv("WORKSPACE"))
tox.cmdline()  # environment is selected by ``TOXENV`` env variable

end
'
'''
        }

      }
    }
  }
}
||||||| merged common ancestors
=======
pipeline {
  agent any
  stages {
    stage('step') {
      steps {
        script {
          sh 'python --version'
        }

      }
    }
  }
}
>>>>>>> Skipping shiningpanda
