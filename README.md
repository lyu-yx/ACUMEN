

## News

## COD-TAX Overview


## ACUMEN Overview




## Experiment Setting

install dependencies in ·requirements.txt·
torch1.9 + cuda10.2 is recomanded


ACUMEN reuslts for _CAMO_, _COD10K_, and _NC4K_ can be found in [ACUMEN results](https://drive.google.com/file/d/1Xywb2vvgiIR8SjSV-guWswSCNsLVvFnF/view?usp=sharing).

Manage the traning and testing dataset like this, Desc is provided by [COD-TAX]() and fixation information can be found from [COD-Rank-Localize-and-Segment](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment).
    
    ACUMEN
        --dataset
            --TestDataset
                --CAMO
                --CHAMELEON
                --COD10K
                --NC4k
            --TrainDataset
                --Desc
                    --overall_description
                    --attribute_description
                    --attribute_contribution
                --Imgs
                --Fix
                --GT
          

put pretrained `ViT-L-14-336px.pt` here. [pretrained ViT](https://drive.google.com/file/d/1Wm9_Dl6M5ETR9qZod3CwWEToMCfDotjg/view?usp=sharing).


    ACUMEN
        --pretrain
            ViT-L-14-336px.pt

## Training
For the training process, run:

    python train_multigpu_noattr.py --config config/codclip_vit_L14@336_noattr_3_1_50.yaml

## Testing / Inference
Put the pretrained checkpoint [here](https://drive.google.com/file/d/1lBMEbeST62KIq4MtJnI9hq19krae0Nxw/view?usp=sharing).

    ACUMEN
        --exp/metapara_noattr_3_1_50
            Net_epoch_best.pth

And run:

    python test.py --config config/codclip_vit_L14@336_noattr_3_1_50.yaml