pipeline {
  agent any
  stages {
    stage('tox') {
      steps {
        script {
          sh '''
python <<end
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
