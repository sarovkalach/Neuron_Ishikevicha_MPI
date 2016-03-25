grep malloc main.c | sed s/\*//g | awk '{print "free("$2");"}'
grep malloc create_add_mass.c | sed s/\*//g | awk '{print "free("$2");"}'
