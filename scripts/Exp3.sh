# This is for reproducing Figure 4.
dataset=IMDB-BINARY
echo "Start exp3 on ${dataset}!!"
python ./Exp3.py --model GST --dataset $dataset
python ./Exp3.py --model GST --dataset $dataset --remove_guo
python ./Exp3.py --model GST --dataset $dataset --retrain
python ./Exp3.py --model GIN --dataset $dataset

dataset=COLLAB
echo "Start exp3 on ${dataset}!!"
python ./Exp3.py --model GST --dataset $dataset
python ./Exp3.py --model GST --dataset $dataset --remove_guo
python ./Exp3.py --model GST --dataset $dataset --retrain