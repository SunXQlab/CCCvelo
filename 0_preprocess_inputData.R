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
cell_file <- files[grep("cellbin",files)]

cellbin_cnt <- read.csv(paste0(data_path,"cellbin_clustered_with_count.csv"))
cellbin_cnt <- cellbin_cnt[,-1]
cellbin_cnt <- t(cellbin_cnt) %>% as.matrix(.)
cellbin_cnt[1:10,1:10]
str(cellbin_cnt)

cell_num <- dim(cellbin_cnt)[2]
gene_num <- dim(cellbin_cnt)[1]

cellbin_gene <- read_csv("data/cellbin_clustered_with_gene.csv") %>% .[,-1] %>% as.data.frame(.)
cellbin_meta <- read_csv("data/cellbin_clustered_with_meta.csv") %>% .[,-1] %>% as.data.frame(.)
cellbin_loc <- read_csv("data/cellbin_clustered_with_loc.csv") %>% .[,-1] %>% as.data.frame(.)
rownames(cellbin_cnt) <- cellbin_gene$`0`
colnames(cellbin_cnt) <- paste0("cell_",seq(cell_num))
str(cellbin_cnt)

rownames(cellbin_meta) <- colnames(cellbin_cnt)
rownames(cellbin_loc) <- colnames(cellbin_cnt)
colnames(cellbin_loc) <- c("x","y")
str(cellbin_meta)

# creat seurat object 
cellbin_seur <- CreateSeuratObject(cellbin_cnt,
                                   meta.data = cellbin_meta, 
                                   assay="Spatial",
                                   min.cells = 20)
cellbin_seur@images$spatial <- cellbin_loc
cellbin_seur$Celltype <- as.factor(cellbin_seur$Celltype)
Idents(cellbin_seur) <- cellbin_seur$Celltype

# filtering
# ct_ind <- as.data.frame(table(seur$new_anno))
# ct <- as.character(ct_ind$Var[which(ct_ind$Freq>50)])
# ind <- which(tmp_cell_types %in% ct)
# seur <- seur[,ind]
# seur = subset(seur, subset = new_anno != "Unknown")
# Idents(seur) <- seur$new_anno

df_loc <- data.frame(x = cellbin_loc$x,y=cellbin_loc$y,
                     celltype = factor(cellbin_meta$Celltype))

plot2 <- ggplot(df_loc, aes(x=x,y=y,colour = celltype)) +
  geom_point(size = 1)
plot2

# select neuronal layer (L2-L6)
Idents(cellbin_seur) <- "Celltype"
neuronal_ct <- c("EX L2/3","EX L5/6","EX L4","EX L6")
neur_meta <- cellbin_meta[which(cellbin_meta$Celltype %in% neuronal_ct), ]
neur_loc <- cellbin_loc[rownames(neur_meta),]

df_loc <- data.frame(x = neur_loc$x,y=neur_loc$y,
                     celltype = factor(neur_meta$Celltype))

plot2 <- ggplot(df_loc, aes(x=x,y=y,colour = celltype)) +
  geom_point(size = 1)
plot2

# select ICGs as target gene
sub_loca <- data.frame(sub_meta$spatial_x, sub_meta$spatial_y)
sub_anno <- data.frame(Barcode=rownames(sub_meta),Cluster=sub_meta$`sim anno`)
rownames(sub_loca) <- rownames(sub_anno)

# creat seurat object
rownames(sub_anno) <- sub_anno$Barcode
ser_obj <- CreateSeuratObject(counts = sub_count, meta.data = sub_anno, assay = 'Spatial')
ser_obj <- SCTransform(ser_obj, assay = 'Spatial')
ser_obj <- FindVariableFeatures(ser_obj, nfeatures = 3000)

## imputation
seed <- 4321
norm.matrix <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
exprMat.Impute <- run_Imputation(exprMat = norm.matrix,use.seed = T,seed = seed)

ser_obj <- ser_obj[VariableFeatures(ser_obj),]
Idents(ser_obj) <- ser_obj@meta.data$Cluster

Databases <- readRDS('./prior_knowledge/Databases.rds')
LRTG_list <- select_LRTG(ser_obj, Databases, log.gc = 0.15, p_val_adj=0.05, pct.ct=0.01, expr.ct = 0.01)

TGs_list <- LRTG_list[["TGs_list"]]
Ligs_expr_list <- LRTG_list[["Ligs_expr_list"]]
Recs_expr_list <- LRTG_list[["Recs_expr_list"]]
# save(TGs_list, Ligs_expr_list, Recs_expr_list, file="conditate_LRTG_list.rda")

write_json(TGs_list, "./bin60_input_with_para25/TGs_list.json", pretty = TRUE, auto_unbox = TRUE)
write_json(Ligs_expr_list, "./bin60_input_with_para25/Ligs_list.json", pretty = TRUE, auto_unbox = TRUE)
write_json(Recs_expr_list, "./bin60_input_with_para25/Recs_list.json", pretty = TRUE, auto_unbox = TRUE)

available_assays <- names(ser_obj@assays)
print("Available Assays: ")
print(available_assays)

if ("SCT" %in% available_assays) {
  df_count <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
} else if ('RNA' %in% available_assays) {  
  df_count <- as.matrix(GetAssayData(ser_obj, "data", "RNA"))
} else {
  # If neither SCT nor RNA is available, use the "spatial" assay
  df_count <- as.matrix(GetAssayData(ser_obj, "data", "Spatial"))
}

write.table(df_count,file=paste0(output_fpath, 'raw_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(exprMat.Impute,file=paste0(output_fpath, 'imputation_expression_mtx.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(sub_anno,file=paste0(output_fpath, 'cell_meta.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
write.table(sub_loca,file=paste0(output_fpath, 'cell_location.csv'),sep = ",",row.names = TRUE,col.names = TRUE)
