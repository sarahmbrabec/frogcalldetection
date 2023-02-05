df <- read.csv("Frogs_MFCCs - Frogs_MFCCs.csv")
df <- subset (df, select = -24)
library(xgboost)
library(caret)  
df[is.na(df)] <- 0
colnames(df) <- c('1','2','3','4','5','6','7','8','9','10','11','12','13','14',
                  '15','16','17','18','19','20','21','22','species')
df$species <- factor(df$species)
species = df$species
label = as.integer(df$species)-1
df$species = NULL

n = nrow(df)
train.index = sample(n,floor(0.75*n))
train.data = as.matrix(df[train.index,])
train.label = label[train.index]
test.data = as.matrix(df[-train.index,])
test.label = label[-train.index]

xgb.train = xgb.DMatrix(data=train.data,label=train.label)
xgb.test = xgb.DMatrix(data=test.data,label=test.label)

num_class = length(levels(species))
params = list(
  booster="gbtree",
  eta=0.001,
  max_depth=5,
  gamma=3,
  subsample=0.75,
  colsample_bytree=1,
  objective="multi:softprob",
  eval_metric="mlogloss",
  num_class=num_class
)

# Train the XGBoost classifer
xgb.fit=xgb.train(
  params=params,
  data=xgb.train,
  nrounds=10000,
  nthreads=1,
  early_stopping_rounds=10,
  watchlist=list(val1=xgb.train,val2=xgb.test),
  verbose=0
)

xgb.pred = predict(xgb.fit,test.data,reshape=T)
xgb.pred = as.data.frame(xgb.pred)
colnames(xgb.pred) = levels(species)

xgb.pred$prediction = apply(xgb.pred,1,function(x) colnames(xgb.pred)[which.max(x)])
xgb.pred$label = levels(species)[test.label+1]

result = sum(xgb.pred$prediction==xgb.pred$label)/nrow(xgb.pred)


