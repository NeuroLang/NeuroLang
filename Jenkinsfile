pipeline {
  agent any
  stages {
    stage('Test') {
      steps {
        script {
          sh ‘python <<end
import urllib, os

url = "https://bitbucket.org/hpk42/tox/raw/default/toxbootstrap.py"
# os.environ['USETOXDEV']="1"  # use tox dev version
d = dict(__file__="toxbootstrap.py")
exec urllib.urlopen(url).read() in d
d["cmdline"](["--recreate"])
end’
        }

      }
    }
  }
}
