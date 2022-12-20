for model in GFT linear-GST
do
    for dataset in IMDB-BINARY PROTEINS COLLAB MNIST CIFAR10
    do
        echo "Start exp1 for ${model} on ${dataset}!!"
        python ./Exp1.py --model $model --dataset $dataset
    done
done