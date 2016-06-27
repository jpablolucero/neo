#!/bin/bash

if test ! -d source -o ! -d include ; then
  echo "*** This script must be run from the top-level directory."
  exit 1
fi

if test ! -f astyle.rc ; then
  echo "*** No style file astyle.rc found."
  exit 1
fi

if test -z "`which astyle`" ; then
  echo "*** No astyle program found."
  echo "***"
  echo "*** You can download astyle from http://astyle.sourceforge.net/"
  echo "*** Note that you will need exactly version 2.04 (no newer or"
  echo "*** older version will yield the correct indentation)."
  exit 1
fi

if test "`astyle --version 2>&1`" != "Artistic Style Version 2.04" ; then
  echo "*** Found a version of astyle different than the required version 2.04."
  exit 1
fi

# collect all header and source files and process them in batches of 50 files
# with up to 10 in parallel
find include source \( -name '*.cc' -o -name '*.h' \) -print | xargs -n 50 -P 10 astyle --options=astyle.rc

