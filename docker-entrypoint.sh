#!/bin/bash
set -e

# Command-line arguments
args="$@"

# Funzione per eseguire il comando principale del container
run_main_command() {
    echo "Running main command: $args"
    exec "$@"
}

# Avvio dell'applicazione Python
start_python_app() {
    echo "Starting Python application..."

    # Avvia l'applicazione Python
    run_main_command python app.py
}

# Avvio dello script
main() {
    # Avvio dell'applicazione Python
    start_python_app "$@"
}

# Esegui la funzione principale
main "$@"
