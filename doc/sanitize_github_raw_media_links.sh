#!/usr/bin/env bash
set -euo pipefail


# Run this script in the source directory:
#     cd source
#     bash ../sanitize_github_raw_media_links.sh


# Recursively scan text files under the current directory, skipping .git
find . -type d -name .git -prune -o -type f -print0 |
while IFS= read -r -d '' file; do
  # Skip binary files
  if ! grep -Iq . "$file"; then
    continue
  fi

  # First: print every match and its replacement
  perl -0ne '
    while (
      m{https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)/?\?raw=true}g
    ) {
      my $old = $&;
      my $new = "https://raw.githubusercontent.com/$1/$2/refs/heads/$3/$4";
      print "$ARGV\n";
      print "  OLD: $old\n";
      print "  NEW: $new\n";
    }
  ' "$file"

  # Then: rewrite the file in place
  perl -0pi -e '
    s{https://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+?)/?\?raw=true}
     {"https://raw.githubusercontent.com/$1/$2/refs/heads/$3/$4"}gex
  ' "$file"
done
