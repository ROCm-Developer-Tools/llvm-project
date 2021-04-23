
#
# build_tables_from_topology_c.sh:
#
# This script is NOT part of the automated clang/tools/offload-arch build system.

# It is provided as example of how one might automate the creation of vendor
# specific input tables pciid2codename.txt and codename2offloadarch.txt
# This script does NOT use pci.ids as input but one could be written to 
# extract information out of that publically available resource. 
#  
# NOTE: When the vendor specific offload-arch tables are updated by vendors,
# either by hand or through some automated process, the responsible vendor
# should upstream their updates to their tables in the vendor directory of
# offload-arch so all future clang builds will have updated information 
# in the offload-arch tool.
# 

topology_source="topology.c"
tmpfile="tmpfile"
pciid2codename_file="pciid2codename.txt"
codename2offloadarch_file="codename2offloadarch.txt"

[ -f $tmpfile ] &&  rm $tmpfile
touch $tmpfile
[ -f $pciid2codename_file ] &&  rm $pciid2codename_file
touch $pciid2codename_file
[ -f $codename2offloadarch_file ] &&  rm $codename2offloadarch_file
touch $codename2offloadarch_file

cat $topology_source | grep CHIP_ | grep 0x | while read -r line ; do 
  chipid=`echo $line | cut -d"{" -f2  | cut -d"," -f1 | cut -d"x" -f2`
  f1=`echo $line | cut -d"{" -f2  | cut -d"," -f2 | xargs`
  f2=`echo $line | cut -d"{" -f2  | cut -d"," -f3 | xargs`
  f3=`echo $line | cut -d"{" -f2  | cut -d"," -f4 | xargs`
  codename_unused=`echo $line | cut -d"{" -f2  | cut -d"," -f5 | xargs`
  CHIP_codename=`echo $line | cut -d"{" -f2  | cut -d"," -f6 | cut -d"}" -f1 | xargs`
  codename=${CHIP_codename#*_}
  echo "$codename gfx$f1$f2$f3" >>$tmpfile
  echo "1002:$chipid 0000 0000 $codename : $codename_unused" >>$pciid2codename_file
done

# Remove duplicat codename2offloadarch entries
cat $tmpfile | sort -u >$codename2offloadarch_file
rm $tmpfile

