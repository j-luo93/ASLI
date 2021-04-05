set -e

function install_dep {
    cd $1
    pip install -r requirements.txt
    pip install -e .
    cd ..
}

install_dep pypheature
install_dep editdistance
install_dep dev_misc 
install_dep .
