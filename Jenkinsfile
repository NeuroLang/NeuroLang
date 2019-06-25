pipeline {
  agent any
  tools {
    python 'anaconda-python-package'
    python 'Anaconda-CPython-3.6'
    python 'Anaconda-CPython-3.7'
  }
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
