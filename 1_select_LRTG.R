library(Seurat)
library(readr)
library(dplyr)
library(ggsci)
library(scales)
library(ggplot2)
library(SeuratWrappers)

rm(list = ls())
gc()

setwd('/home/yll/velocity_methods/01_analysis/apply_in_prostate/area_4000x6000_5000x7000_input_v2/')
source('/home/yll/velocity_methods/01_analysis/apply_in_stereo_cortex/R/preprocess_code.R')

# load data
data_path <- '/home/yll/velocity_methods/01_analysis/apply_in_prostate/data/area_4000x6000_5000x7000/'
ser_obj <- readRDS(paste0(data_path,'sub_ser_obj.rds'))

ser_obj@meta.data$Cluster <- ser_obj@meta.data$new_celltype
Idents(ser_obj) <- ser_obj@meta.data$Cluster

## imputation
seed <- 4321
norm.matrix <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
exprMat.Impute <- run_Imputation(exprMat = norm.matrix,use.seed = T,seed = seed)

sub_anno <- data.frame(Barcode=colnames(ser_obj),Cluster=ser_obj$Cluster)
sub_loca <- data.frame(x=ser_obj$center_x,y=ser_obj$center_y)

# load prior databse
Databases <- readRDS('/home/yll/velocity_methods/01_analysis/prior_knowledge/Databases.rds')
quan.cutoff <- 0.98
Databases <- Databases
Databases$RecTF.DB <- Databases$RecTF.DB %>%
  .[.$score > quantile(.$score, quan.cutoff),] %>%
  dplyr::distinct(source, target)
Databases$LigRec.DB <- Databases$LigRec.DB %>%
  dplyr::distinct(source, target) %>%
  dplyr::filter(target %in% Databases$RecTF.DB$source)
Databases$TFTG.DB <- Databases$TFTG.DB %>%
  dplyr::distinct(source, target) %>%
  dplyr::filter(source %in% Databases$RecTF.DB$target)

LRTG_list <- select_LRTG(bin60_seur, Databases, log.gc = 0.4, p_val_adj=0.05,
                         pct.ct=0.05, expr.ct = 0.5)
TGs_list <- LRTG_list[["TGs_list"]]
Ligs_expr_list <- LRTG_list[["Ligs_expr_list"]]
Recs_expr_list <- LRTG_list[["Recs_expr_list"]]

## save results
project_name = your_project_name
output_fpath <- paste0(getwd(), '/Input/your_project_name')

if(!dir.exists(output_fpath)){
    dir.create(output_fpath)
}

write_json(TGs_list, path=paste0(output_fpath,"TGs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Ligs_expr_list, path=paste0(output_fpath,"Ligs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Recs_expr_list, path=paste0(output_fpath,"Recs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Databases, path=paste0(output_fpath,"Databases.json"), pretty = TRUE, auto_unbox = TRUE)

df_count <- as.matrix(GetAssayData(ser_obj, "data", "Spatial"))
rownames(exprMat.Impute) = rownames(df_count)
df_count = df_count[rownames(exprMat.Impute),]
df_count = t(df_count)
exprMat.Impute = t(exprMat.Impute)


write.table(df_count,file=paste0(output_fpath, 'raw_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(exprMat.Impute,file=paste0(output_fpath, 'imputation_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(sub_anno,file=paste0(output_fpath, 'cell_meta.csv'),sep = ",",row.names = FALSE,col.names = TRUE)
write.table(sub_loca,file=paste0(output_fpath, 'cell_location.csv'),sep = ",",row.names = TRUE,col.names = TRUE)


