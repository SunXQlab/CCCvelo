library(Seurat)
library(readr)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(stMLnet)
library(SeuratWrappers)

rm(list = ls())
gc()

setwd("/home/yll/velocity_methods/01_analysis/apply_in_stereo_cortex")

source('/home/yll/velocity_methods/01_analysis/apply_in_stereo_cortex/R/preprocess_code.R')

#################
# load datasets #
#################

data_path <- paste0(getwd(),'/data/')
files <- list.files(data_path)
files <- files[grep(".csv",files)]
bin60_file <- files[grep("bin60",files)]

bin60_cnt <- read.csv(paste0(data_path,"bin60_clustered_with_count.csv")) %>% .[,-1]
bin60_gene <- read_csv(paste0(data_path,"bin60_clustered_with_gene.csv")) %>% .[,-1] %>% as.data.frame(.)
bin60_meta <- read_csv(paste0(data_path,"bin60_clustered_with_meta.csv")) %>% .[,-1] %>% as.data.frame(.)
bin60_loc <- read_csv(paste0(data_path,"bin60_clustered_with_loc.csv")) %>% .[,-1] %>% as.data.frame(.)

bin60_cnt <- t(bin60_cnt) %>% as.matrix(.)
gene_num <- dim(bin60_cnt)[1]
cell_num <- dim(bin60_cnt)[2]

rownames(bin60_cnt) <- toupper(bin60_gene$`0`)
colnames(bin60_cnt) <- paste0("bin60_",seq(cell_num))
str(bin60_cnt)

bin60_meta$scc_anno <- gsub("/","",bin60_meta$scc_anno)
rownames(bin60_meta) <- colnames(bin60_cnt)
rownames(bin60_loc) <- colnames(bin60_cnt)
colnames(bin60_loc) <- c("x","y")
str(bin60_meta)

# creat seurat object 
bin60_seur <- CreateSeuratObject(bin60_cnt,
                                 meta.data = bin60_meta, 
                                 assay="Spatial",
                                 min.cells = 20)

# preprocess
bin60_seur <- SCTransform(bin60_seur, assay = 'Spatial')
Idents(bin60_seur) <- bin60_seur@meta.data$scc_anno
bin60_seur@meta.data$Cluster <- bin60_seur@meta.data$scc_anno

Databases <- readRDS('../newDatabase/Databases.rds')
LRTG_list <- select_LRTG(bin60_seur, Databases, log.gc = 0.25, p_val_adj=0.05,
                         pct.ct=0.01, expr.ct = 0.1)
TGs_list <- LRTG_list[["TGs_list"]]
Ligs_expr_list <- LRTG_list[["Ligs_expr_list"]]
Recs_expr_list <- LRTG_list[["Recs_expr_list"]]

## imputation
seed <- 4321
norm.matrix <- as.matrix(GetAssayData(bin60_seur, "data", "SCT"))
exprMat.Impute <- run_Imputation(exprMat = norm.matrix,use.seed = T,seed = seed)

## save results

output_fpath <- paste0(getwd(), '/bin60_input_test/')

write_json(TGs_list, file=paste0(output_fpath,"TGs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Ligs_expr_list, file=paste0(output_fpath,"Ligs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Recs_expr_list, file=paste0(output_fpath,"Recs_list.json"), pretty = TRUE, auto_unbox = TRUE)
write_json(Databases, file=paste0(output_fpath,"Databases.json"), pretty = TRUE, auto_unbox = TRUE)

df_count <- as.matrix(GetAssayData(bin60_seur, "data", "Spatial"))
rownames(exprMat.Impute) = rownames(df_count)
df_count = df_count[rownames(exprMat.Impute),]
df_count = t(df_count)
exprMat.Impute = t(exprMat.Impute)
  
write.table(df_count,file=paste0(output_fpath, 'raw_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(exprMat.Impute,file=paste0(output_fpath, 'imputation_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(bin60_meta,file=paste0(output_fpath, 'cell_meta.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(bin60_loc,file=paste0(output_fpath, 'cell_location.csv'),sep = ",",row.names = TRUE,col.names = TRUE)

