pipeline {
  agent any
  tools {
    PythonInstallation 'anaconda-python-package'
    PythonInstallation 'Anaconda-CPython-3.6'
    PythonInstallation 'Anaconda-CPython-3.7'
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
