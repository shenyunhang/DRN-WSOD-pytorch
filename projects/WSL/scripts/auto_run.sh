
set -x

CMD="$@"
echo ${CMD}
until ${CMD}
do
	echo "Try again"
	echo ${CMD}
done
