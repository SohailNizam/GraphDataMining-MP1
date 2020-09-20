#############################################
#### Benjamin Risk
#### Edited by Sohail Nizam
##############################################

library(R.matlab)
library(lme4) 
library(lmerTest)
library(tidyr)
library(ggplot2)
library(gridExtra)
library(matlab)
library(fields)
library(tidyr)
library(ggplot2)
library(wesanderson)
library(ggpubr)
library(glasso)
library(RcppCNPy)


# Change working directory depending on whether running on cluster or locally:
setwd("~/Desktop")

source('./OptimalSMS_rsfMRI/Programs/Functions/fconn_fun.R')
alldat = readMat('CorrSDpower264_9p.mat')
alldat_tf = readMat('./OptimalSMS_rsfMRI/Results/CorrSDpower264_9p_tf.mat')




######################
######################
######################
# Communities
# note: Use node2 because took lower triangle
# use 1:13 for subcortical
# use 1:17 for subcortical + cerebellum
# use 18:47 for sensory/motor
# use 80:137 for DMN
subcortical = 1:13
cerebellum = 14:17
sensomotorH = 18:47
sensomotorM = 48:52
cingulo = 53:66
auditory = 67:79
dmn = 80:137
memory = 138:142
visual = 143:173
fptaskcontr=174:198
salience=199:216
ventralatt=217:225
dorsatt=226:236
uncertain=237:264
labs2=c("Subc","Cer","SomH","SomM","CO-C","Aud","DMN","Mem","Vis","FP-C","Sal","VenA","DorA","Uncr")

# correspond to sorted indices:
labelsLong=c(rep("Subc",13),rep("Cer",4),rep("SomH",30),rep("SomM",5),rep("CO-TC",14),rep('Auditory',13),rep('Default Mode',58),rep('Memory',5),rep('Visual',31),rep('FP-TC',15),rep('Salience',18),rep("VentralAtt",9),rep("DorsalAtt",11),rep("Uncertain",38))

varnames = c("SB.3.3.mm","SB.2.mm", "MB.2", "MB.3","MB.4","MB.6","MB.8","MB.9","MB.12")
varnamesplots = c("1-3.3","1-2", "2", "3","4","6","8","9","12")
varnamesplots2 = c("SB 3.3mm","SB 2mm", "MB 2", "MB 3","MB 4","MB 6","MB 8","MB 9","MB 12")

################
###############

dim(alldat$allmb.cor.srt)

nsubject = dim(alldat$allmb.cor.srt)[4]
node1 = c(1:264)%*%t(rep(1,264))
node2 = rep(1,264)%*%t(c(1:264))


allmb.cor.srt = alldat$allmb.cor.srt  ##Contains communities sorted so that subcortical is first and uncertain last
allmb.gg.srt = alldat$allmb.gg.srt
allmb.sd.srt = alldat$allmb.sdsd.srt


# Create mean correlation matrices (skip fisher transformation)
mean.cor = apply(allmb.cor.srt,c(1,2,3),mean,na.rm=TRUE)
dim(mean.cor)


# Graphical lasso on mean corr matrices
# to obtain precision matrix estimate
#mb1_prec <- glasso(mean.cor[,,1], rho = .02)$wi
mb2_prec <- glasso(mean.cor[,,2], rho = .0025)$wi
mb3_prec <- glasso(mean.cor[,,3], rho = .0025)$wi
mb4_prec <- glasso(mean.cor[,,4], rho = .0025)$wi
mb5_prec <- glasso(mean.cor[,,5], rho = .0025)$wi
mb6_prec <- glasso(mean.cor[,,6], rho = .0025)$wi
mb7_prec <- glasso(mean.cor[,,7], rho = .0025)$wi
mb8_prec <- glasso(mean.cor[,,8], rho = .0025)$wi
mb9_prec <- glasso(mean.cor[,,9], rho = .0025)$wi

# Calculate sparsity in prec matrices
# rho=.0025 gives avg of 15.1% sparsity
# *mb1 omitted. glasso giving us trouble on it
sparsity_vec = c(
  #sum(colSums(mb1_prec == 0)) / (264*264),
  sum(colSums(mb2_prec == 0)) / (264*264),
  sum(colSums(mb3_prec == 0)) / (264*264),
  sum(colSums(mb4_prec == 0)) / (264*264),
  sum(colSums(mb5_prec == 0)) / (264*264),
  sum(colSums(mb6_prec == 0)) / (264*264),
  sum(colSums(mb7_prec == 0)) / (264*264),
  sum(colSums(mb8_prec == 0)) / (264*264),
  sum(colSums(mb9_prec == 0)) / (264*264)
)

# save each as a python numpy array
npySave("mb2.npy", mb2_prec)
npySave("mb3.npy", mb3_prec)
npySave("mb4.npy", mb4_prec)
npySave("mb5.npy", mb5_prec)
npySave("mb6.npy", mb6_prec)
npySave("mb7.npy", mb7_prec)
npySave("mb8.npy", mb8_prec)
npySave("mb9.npy", mb9_prec)
