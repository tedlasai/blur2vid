# Motion Metrics
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
np.float = np.float64
np.int = np.int_
import os
from cdfvd import fvd
from skimage.metrics import structural_similarity
import torch
import lpips
#from DISTS_pytorch import DISTS
#import colour as c
#from torchmetrics.image.fid import FrechetInceptionDistance
import torch.nn.functional as F
from epe_metric import compute_bidirectional_epe as epe
import pdb
import multiprocessing
import cv2
import glob
# init
# dataDir = 'BaistCroppedOutput' # 'dataGoPro' # 
# gtDir = 'gt_subset' #'GT' # 
# methodDirs = ['deblurred', 'animation-from-blur', ] #['Favaro','MotionETR','Ours','GOPROGeneralize']  # 
# fType = '.mp4'
# depth = 8
# resFile = './resultsBaist20250521.npy'#resultsGoPro20250520.npy'# 

# patchDim = 32 #64 #  
# pixMax = 1.0

# nMets = 7 # new results: scoreFVD, scorePWPSNR, scoreEPE, scorePatchSSIM, scorePatchLPIPS, scorePSNR
# compute = True # if False, load previously computed
# eps = 1e-8 

dataDir = 'GOPROResultsImages' # 'dataBaist' #
gtDir = 'GT' #'gt' #
methodDirs = ['Jin','MotionETR','Ours']  #'GOPROGeneralize',# ['animation-from-blur'] #
depth = 8
resFile = 'resultsGoPro20250521.npy'# './resultsBaist20250521.npy'#
patchDim = 40 #32 #
pixMax = 1.0
nMets = 7 # new results: scoreFVD, scorePWPSNR, scoreEPE, scorePatchSSIM, scorePatchLPIPS, scorePSNR
compute = False # if False, load previously computed
eps = 1e-8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use 'spawn' to avoid CUDA context issues
multiprocessing.freeze_support()              # on Windows
multiprocessing.set_start_method('spawn', force=True)

def read_pngs_to_array(path):
    """Read all PNGs in `path`, sort them by filename, convert BGR→RGB, and stack into an np.ndarray."""
    return np.stack([
        cv2.imread(f, cv2.IMREAD_UNCHANGED)[..., ::-1]
        for f in sorted(glob.glob(f"{path}/*.png"))
    ])
def compute_method(results_local, methodDir, files, countMethod):

    fnLPIPS = lpips.LPIPS(net='alex').to(device)
    #fnDISTS = DISTS().to(device)
    fnFVD = fvd.cdfvd(model='videomae', device=device)

    countFile = -1
    for file in files:
        countFile+=1

        # pull frames from MP4
        pathMethod = os.path.join(dataDir, methodDir, file)
        framesMethod = np.clip(read_pngs_to_array(pathMethod).astype(np.float32) / (2**depth-1),0,1)
        pathGT = os.path.join(dataDir, gtDir, file)
        framesGT = np.clip(read_pngs_to_array(pathGT).astype(np.float32) / (2**depth-1),0,1)

        #make sure the GT and method have the same shape
        assert framesGT.shape == framesMethod.shape, f"GT shape {framesGT.shape} does not match method shape {framesMethod.shape} for file {file}"

        # video metrics

        # vmaf
        #scoreVMAF = callVMAF(pathGT, pathMethod)

        # epe - we have to change to tensors here
        framesMethodTensor = torch.from_numpy(framesMethod)
        framesGTtensor = torch.from_numpy(framesGT)
        scoreEPE = epe(framesMethodTensor[0,:,:,:], framesMethodTensor[-1,:,:,:], framesGTtensor[0,:,:,:], framesGTtensor[-1,:,:,:], per_pixel_mode=True).cpu().detach().numpy()

        # motion blur baseline
        blurryGT = np.mean(framesGT ** 2.2,axis=0) ** (1/2.2)
        blurryMethod = np.mean(framesMethod ** 2.2,axis=0) ** (1/2.2)
        # MSE -> PSNR
        mapBlurryMSE = (blurryGT - blurryMethod)**2
        scoreBlurryMSE = np.mean(mapBlurryMSE)
        scoreBlurryPSNR = (10 * np.log10(pixMax**2 / scoreBlurryMSE))            

        # fvd
        #scoreFVD = fnFVD.compute_fvd(real_videos=(np.expand_dims(framesGT, axis=0)*(2**depth-1)).astype(np.uint8), fake_videos=(np.expand_dims(framesMethod, axis=0)*(2**depth-1)).astype(np.uint8))
        framesGTfvd = np.expand_dims((framesGT * (2**depth-1)).astype(np.uint8), axis=0)
        fnFVD.add_real_stats(framesGTfvd)
        framesMethodFVD = np.expand_dims((framesMethod * (2**depth-1)).astype(np.uint8), axis=0)
        fnFVD.add_fake_stats(framesMethodFVD)

        # loop directions
        framesMSE = np.stack((framesGT,framesGT)) # pre allocate array for directional PSNR maps
        countDirect = -1
        for direction in directions:
            countDirect = countDirect+1
            order = direction

            # loop frames + image level metrics
            countFrames = -1
            for i in order:
                countFrames+=1

                frameMethod = framesMethod[i,:,:,:] # method frames can be re-ordered
                frameGT =  framesGT[countFrames,:,:,:]

                #assert patch size is divisible by image size
                rows, cols, ch = frameGT.shape
                assert rows % patchDim == 0, f"rows {rows} is not divisible by patchDim {patchDim}"
                assert cols % patchDim == 0, f"cols {cols} is not divisible by patchDim {patchDim}"

                rPatch = np.ceil(rows/patchDim)
                cPatch = np.ceil(cols/patchDim)

                # LPIPS
                #pdb.set_trace()
                methodTensor = (torch.from_numpy(np.moveaxis(frameMethod, -1, 0)).unsqueeze(0) * 2 - 1).to(device)
                gtTensor = (torch.from_numpy(np.moveaxis(frameGT, -1, 0)).unsqueeze(0) * 2 - 1).to(device)
                #scoreLPIPS = fnLPIPS(gtTensor, methodTensor).squeeze(0,1,2).cpu().detach().numpy()[0]

                # FID                 
                #fnFID.update((gtTensor * (2**depth - 1)).to(torch.uint8), real=True)
                #fnFID.update((methodTensor * (2**depth - 1)).to(torch.uint8), real=False)               

                # DISTS
                #scoreDISTS = fnDISTS(gtTensor.to(torch.float), methodTensor.to(torch.float), require_grad=True, batch_average=True).cpu().detach().numpy()                    

                # compute ssim
                #scoreSSIM = structural_similarity(frameGT, frameMethod, data_range=pixMax, channel_axis=2)

                # compute DE 2000
                #frameMethodXYZ = c.RGB_to_XYZ(frameMethod, c.models.RGB_COLOURSPACE_sRGB, apply_cctf_decoding=True)
                #frameMethodLAB = c.XYZ_to_Lab(frameMethodXYZ)
                #frameGTXYZ = c.RGB_to_XYZ(frameGT, c.models.RGB_COLOURSPACE_sRGB, apply_cctf_decoding=True)
                #frameGTLAB = c.XYZ_to_Lab(frameGTXYZ)
                #mapDE2000 = c.delta_E(frameGTLAB, frameMethodLAB, method='CIE 2000')
                #scoreDE2000 = np.mean(mapDE2000)

                # MSE
                mapMSE = (frameGT - frameMethod)**2
                scoreMSE = np.mean(mapMSE)

                # PSNR
                framesMSE[countDirect,countFrames,:,:,:] = mapMSE
                #framesPSNR[countDirect,countFrames,:,:,:] = np.clip((10 * np.log10(pixMax**2 / np.clip(mapMSE,a_min=1e-10,a_max=None))),0,100)
                scorePSNR = (10 * np.log10(pixMax**2 / scoreMSE))

                #for l in range(ch):

                    # channel-wise metrics
                    #chanFrameMethod = frameMethod[:,:,l]
                    #chanFrameGT = frameGT[:,:,l]

                # loop patches rows 
                for j in range(int(rPatch)):

                    # loop patches cols + patch level metrics
                    for k in range(int(cPatch)):

                        startR = j*patchDim
                        startC = k*patchDim
                        endR = j*patchDim+patchDim
                        endC = k*patchDim+patchDim

                        if endR > rows:
                            endR = rows                           
                        else:
                            pass

                        if endC > cols:
                            endC = cols
                        else:
                            pass
                            
                        # patch metrics
                        #patchMSE = np.mean(mapMSE[startR:endR,startC:endC,:])
                        #scorePatchPSNR = np.clip((10 * np.log10(pixMax**2 / patchMSE)),0,100)
                        if dataDir == 'BaistCroppedOutput':
                            patchGtTensor = F.interpolate(gtTensor[:,:,startR:endR,startC:endC], scale_factor=2.0, mode='bilinear', align_corners=False)
                            patchMethodTensor = F.interpolate(methodTensor[:,:,startR:endR,startC:endC], scale_factor=2.0, mode='bilinear', align_corners=False)
                            scorePatchLPIPS = fnLPIPS(patchGtTensor, patchMethodTensor).squeeze(0,1,2).cpu().detach().numpy()[0]                                
                        else:
                            scorePatchLPIPS = fnLPIPS(gtTensor[:,:,startR:endR,startC:endC], methodTensor[:,:,startR:endR,startC:endC]).squeeze(0,1,2).cpu().detach().numpy()[0]                                
                        scorePatchSSIM = structural_similarity(frameGT[startR:endR,startC:endC,:], frameMethod[startR:endR,startC:endC,:], data_range=pixMax, channel_axis=2)
                        #scorePatchDISTS = fnDISTS(gtTensor[:,:,startR:endR,startC:endC].to(torch.float), methodTensor[:,:,startR:endR,startC:endC].to(torch.float), require_grad=True, batch_average=True).cpu().detach().numpy()                            
                        #scorePatchDE2000 = np.mean(mapDE2000[startR:endR,startC:endC])

                        # i: frame number, j: patch row, k: patch col
                        #results[countMethod,countFile,countDirect,i,j,k,3:] = [scoreEPE, scoreBlurryPSNR, scoreLPIPS, scoreDISTS, scoreSSIM, scoreDE2000, scorePSNR, scorePatchPSNR, scorePatchSSIM, scorePatchLPIPS, scorePatchDISTS, scorePatchDE2000]
                        results_local[countMethod,countFile,countDirect,i,j,k,2:] = [scoreEPE, scoreBlurryPSNR, scorePatchSSIM, scorePatchLPIPS, scorePSNR]                            
                        print('Method: ', methodDir, ' File: ', file, ' Frame: ', str(i), ' PSNR: ', scorePSNR,  end='\r')
                        
                        #print('VMAF: ', str(scoreVMAF), ' FVD: ', str(scoreFVD), ' LPIPS: ', str(scoreLPIPS), ' FID: ', str(scoreFID), ' DISTS: ', str(scoreDISTS), ' SSIM: ', str(scoreSSIM), ' DE2000: ', str(scoreDE2000), ' PSNR: ', str(scorePSNR), ' Patch PSNR: ', str(scorePatchPSNR), end='\r')
        #pdb.set_trace()
        scorePWPSNR = (10 * np.log10(pixMax**2 / np.mean(np.min(np.mean(framesMSE, axis=(1)),axis=0)))) # take max pixel wise PSNR per direction, average over image dims 
        #print('Method: ', methodDir, ' File: ', file, ' Frame: ', str(i), ' PWPSNR: ', scorePWPSNR,  end='\n')
        #scorePWPSNR = np.clip((10 * np.log10(pixMax**2 / np.mean(np.min(framesPSNR, axis=0),axis=(1,2,3)))),0,100) # take max pixel wise PSNR per direction, average over image dims 
        results_local[countMethod,countFile,:,:,:,:,1] = np.tile(scorePWPSNR, results_local.shape[2:-1])#np.broadcast_to(scorePWPSNR[:, np.newaxis, np.newaxis], results.shape[3:-1])
        np.save(resFile, results_local) # save part of the way through the loop ..

    #scoreFID = fnFID.compute().cpu().detach().numpy()
    #fnFID.reset()
    #results[countMethod,:,:,:,:,:,0] = np.tile(scoreFID, results.shape[1:-1])
    scoreFVD = fnFVD.compute_fvd_from_stats()
    fnFVD.empty_real_stats()
    fnFVD.empty_fake_stats()
    results_local[countMethod,:,:,:,:,:,0] = np.tile(scoreFVD, results_local.shape[1:-1])
    print('Results computed .. analyzing ..')

    return results_local

        
    # init results matrix 
path = os.path.join(dataDir, gtDir)
clipDirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
files = []
if dataDir == 'BaistCroppedOutput':
    extraFknDir = 'blur'
else:
    extraFknDir = ''
for clipDir in clipDirs:
    path = os.path.join(dataDir, gtDir, clipDir, extraFknDir)
    files = files + [os.path.join(clipDir,extraFknDir,name) for name in os.listdir(path)]
files = sorted(files)
path = os.path.join(dataDir, methodDirs[0], files[0])
testFileGT = read_pngs_to_array(path)
frams,rows,cols,ch = testFileGT.shape
framRange = [i for i in range(frams)]
directions = [framRange, framRange[::-1]]

#loop through all methods and make sure they all have the same directory structure and same number of files
for methodDir in methodDirs:
    path = os.path.join(dataDir, methodDir)
    clipDirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    filesMethod = []
    for clipDir in clipDirs:
        path = os.path.join(dataDir, methodDir, clipDir, extraFknDir)
        filesMethod = filesMethod + [os.path.join(clipDir,extraFknDir,name) for name in os.listdir(path)]
    filesMethod = sorted(filesMethod)
    print('Method: ', methodDir, ' Number of files: ', len(filesMethod))
    assert len(files) == len(filesMethod), f"Number of files in {methodDir} does not match GT number of files"
    assert files == filesMethod, f"Files in {methodDir} do not match GT files"


def main():

    results = np.zeros((len(methodDirs),len(files),len(directions),frams,int(np.ceil(rows/patchDim)),int(np.ceil(cols/patchDim)),nMets))

    if compute:
        
        # loop methods + compute dataset level metrics (after nested for loops)
        import multiprocessing as mp
        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(mp_context=ctx, max_workers=len(methodDirs)) as executor:
            # submit one job per method
            futures = {
                executor.submit(compute_method, np.copy(results), md, files, idx): idx
                for idx, md in enumerate(methodDirs)
            }
            # collect and merge results as they finish
            for fut in as_completed(futures):
                idx = futures[fut]
                res_local = fut.result()
                results[idx] = res_local[idx]


    else:

        results = np.load(resFile)

    np.save(resFile, results)
    # analyze

    # new results: scoreFID, scoreFVD, scorePWPSNR, scoreEPE, scoreLPIPS, scoreDISTS, scoreSSIM, scoreDE2000, scorePSNR, scorePatchPSNR, scorePatchSSIM, scorePatchLPIPS, scorePatchDISTS, scorePatchDE2000
    upMetrics = [1,3,4,6]


    # 0508 results: scoreFID, scoreFVD, scoreLPIPS, scoreDISTS, scoreSSIM, scoreDE2000, scorePSNR, scorePatchPSNR, scorePatchSSIM, scorePatchLPIPS, scorePatchDISTS, scorePatchDE2000
    #upMetrics = [4,6,7,8] # PSNR, SSIM, Patch PSNR, Patch SSIM
    print("Results shape 1: ", results.shape)
    forwardBackwardResults = np.mean(results,axis=(3))
    #print("Results shape 2: ", forwardResults.shape)
    maxDirResults = np.max(forwardBackwardResults,axis=(2))
    minDirResults = np.min(forwardBackwardResults,axis=(2))
    bestDirResults = minDirResults
    #pdb.set_trace()
    bestDirResults[:,:,:,:,upMetrics] = maxDirResults[:,:,:,:,upMetrics]
    import pdb
    #pdb.set_trace()

    meanResults = bestDirResults.mean(axis=(1, 2, 3))  # Shape becomes (3, 6)
    meanResultsT = meanResults.T

    '''
    maxDirResults = np.max(results,axis=2)
    minDirResults = np.min(results,axis=2)
    bestDirResults = minDirResults
    bestDirResults[:,:,:,:,:,upMetrics] = maxDirResults[:,:,:,:,:,upMetrics]
    meanResults = bestDirResults.mean(axis=(1, 2, 3, 4))  # Shape becomes (3, 6)
    meanResultsT = meanResults.T
    '''

    #
    #meanResults = forwardResults.mean(axis=(1, 2, 3, 4))  # Shape becomes (3, 6)
    #meanResultsT = meanResults.T

    # print latex table
    method_labels = methodDirs

    # results 0508: scoreLPIPS, scoreDISTS, scoreSSIM, scoreDE2000, scorePSNR, scorePatchPSNR, scorePatchSSIM, scorePatchLPIPS, scorePatchDISTS, scoreFID, scoreFVD
    # metric_labels = ["FID $\downarrow$","FVD $\downarrow$","LPIPS $\downarrow$", "DISTS $\downarrow$", "SSIM $\downarrow$", "DE2000 $\downarrow$", "PSNR $\downarrow$", "Patch PSNR $\downarrow$", "Patch SSIM $\downarrow$",  "Patch LPIPS $\downarrow$", "Patch DISTS $\downarrow$", "Patch DE2000 $\downarrow$"]
    # results 0517: 
    # metric_labels = ["FID $\downarrow$","FVD $\downarrow$","PWPSNR $\downarrow$","EPE $\downarrow$","BlurryPSNR $\downarrow$", "LPIPS $\downarrow$", "DISTS $\downarrow$", "SSIM $\downarrow$", "DE2000 $\downarrow$", "PSNR $\downarrow$", "Patch PSNR $\downarrow$", "Patch SSIM $\downarrow$",  "Patch LPIPS $\downarrow$", "Patch DISTS $\downarrow$", "Patch DE2000 $\downarrow$"]

    # results 0518:
    metric_labels = ["FVD $\downarrow$","PWPSNR $\downarrow$","EPE $\downarrow$","BlurryPSNR $\downarrow$","Patch SSIM $\downarrow$","Patch LPIPS $\downarrow$", "PSNR $\downarrow$"]

    # appropriate for results 0507
    #metric_labels = ["FID $\downarrow$", "FVD $\downarrow$", "LPIPS $\downarrow$", "DISTS $\downarrow$", "SSIM $\downarrow$", "DE2000 $\downarrow$", "PSNR $\downarrow$"]

    latex_table = "\\begin{tabular}{l" + "c" * len(method_labels) + "}\n"
    latex_table += "Metric & " + " & ".join(method_labels) + " \\\\\n"
    latex_table += "\\hline\n"

    for metric, row in zip(metric_labels, meanResultsT):
        row_values = " & ".join(f"{v:.4f}" for v in row)
        latex_table += f"{metric} & {row_values} \\\\\n"

    latex_table += "\\end{tabular}"
    print(latex_table)

if __name__ == '__main__':
    main()
