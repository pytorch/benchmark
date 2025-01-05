target_hour=22
target_min=00
while true
do
    current_hour=$(date +%H)
    current_min=$(date +%M)
    if [ $current_hour -eq $target_hour ] && [ $current_min -eq $target_min ] ; then
        echo "Cron job started at $(date)"
        sh cron_script.sh > local_cron_log 2>local_cron_err
        echo "Cron job executed at $(date)"
    fi
    sleep 60
done
