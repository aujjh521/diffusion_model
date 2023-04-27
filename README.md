# diffusion_model
#注意事項
這邊是先參考一篇教學從頭開始建一個簡單的stable diffusion
- ref: https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch
- diffusion model train起來很慢,所以傳到git的是已經先在自己電腦train好的weight
- 如果要重train,要把程式裡面註解的部分解開 (包含資料下載code + training code)

## 20230423
- cpe 的部分直接用之前soc自創的假資料,在dataset的地方強制變成28*28的image
- 為了方便直接套用原本教學的範例code, image為杜只能先一維, 所以先只拿Tx來看
- 初步結果看起來生成的影像有點怪,還需進一步確認