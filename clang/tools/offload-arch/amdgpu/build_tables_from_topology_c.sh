


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

