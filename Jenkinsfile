pipeline {
  agent any
  tool name: 'Anaconda-CPython-3.6', type: 'jenkins.plugins.shiningpanda.tools.PythonInstallation'
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
