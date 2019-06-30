pipeline {
  agent any
  stages {
    stage('step') {
      steps {
        script {
          sh 'python <<<end
import tox

os.chdir(os.getenv("WORKSPACE"))
tox.cmdline()  # environment is selected by ``TOXENV`` env variable

end
'
        }

      }
    }
  }
}
