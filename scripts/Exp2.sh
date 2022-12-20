# This is for reproducing Figure 3.
for dataset in IMDB-BINARY PROTEINS COLLAB MNIST CIFAR10
do
    echo "Start exp2 on ${dataset}!!"
    python ./Exp2.py --model GST --dataset $dataset
    python ./Exp2.py --model GST --dataset $dataset --remove_guo
    python ./Exp2.py --model GST --dataset $dataset --retrain
    python ./Exp2.py --model linear-GST --dataset $dataset
    python ./Exp2.py --model linear-GST --dataset $dataset --remove_guo
done
