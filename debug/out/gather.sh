#!/bin/bash

echo "Gathering all pages."
cat header.html page*.html footer.html > all.html
