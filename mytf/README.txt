shuf data.csv -o shuffled_data.csv
split -l 32000 shuffled_data.csv shuf_split