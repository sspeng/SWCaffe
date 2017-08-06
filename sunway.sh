sudo pptpsetup --create vpn --server 58.215.62.130 --username fangjr --password 123456 --encrypt --start
sudo ip route replace default dev ppp0
