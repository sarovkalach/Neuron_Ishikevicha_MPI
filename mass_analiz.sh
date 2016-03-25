grep malloc main.c | sed s/*/\ /g | sed s/\(/\ /g   | awk '{print $2"    "$4"    "$6}'
grep malloc create_add_mass.c | sed s/*/\ /g | sed s/\(/\ /g   | awk '{print $2"    "$4"    "$6}'
