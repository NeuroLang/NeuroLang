pipeline {
  agent any
  stages {
    stage('tox') {
      steps {
        script {
          sh '''
python <<end
import os
import tox

os.chdir(os.getenv("WORKSPACE"))
tox.cmdline()
end
          '''
        }

      }
    }
  }
}
