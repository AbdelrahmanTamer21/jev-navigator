#!/bin/sh
# A stand-in LLM: ignores its prompt and prints the next line of the replies file given as $1.
cat > /dev/null
head -n 1 "$1"
tail -n +2 "$1" > "$1.rest" && mv "$1.rest" "$1"
