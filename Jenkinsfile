pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        script {
          sh ‘python <<end
              import urllib, os

              url = "https://bitbucket.org/hpk42/tox/raw/default/toxbootstrap.py"
              d = dict(__file__="toxbootstrap.py")
              exec urllib.urlopen(url).read() in d
              d["cmdline"](["--recreate"])
              end’
        }

      }
    }
  }
}
