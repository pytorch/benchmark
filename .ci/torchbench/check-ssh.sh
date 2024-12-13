#!/bin/bash

# Loop to continuously check the output of `who`
while true; do
    # Check if `who` returns non-empty output
    if who | grep -q '.'; then
        echo "User(s) logged in. Sleeping for 100 seconds..."
        sleep 100
    else
        echo "No users logged in. Exiting..."
        exit 0
    fi
done
