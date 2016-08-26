#!/bin/bash

echo "Gathering all pages."
cat header.html page*.xhtml footer.html > all.html
