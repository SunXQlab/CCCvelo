select_LRTG <- function(ser_obj, Databases, log.gc, p_val_adj, pct.ct, expr.ct){

  # select Target gene
  st_markers <- lapply(unique(ser_obj$Cluster),function(clu){

    markers <- FindMarkers(ser_obj, ident.1 = clu,logfc.threshold = log.gc, min.pct = pct.ct)
    markers$ident.1 <- clu
    markers$gene <- rownames(markers)
    markers

  }) %>% do.call('rbind',.) %>% as.data.frame()
  table(st_markers$p_val_adj <= p_val_adj,st_markers$ident.1)
  df_markers <- st_markers[st_markers$p_val_adj<=p_val_adj,]
  ICGs_list <- split(df_markers$gene,df_markers$ident.1)
  str(ICGs_list)


  ligs_in_db <- Databases$LigRec.DB$source %>% unique()
  ligs_in_db <- intersect(ligs_in_db, rownames(ser_obj))
  recs_in_db <- Databases$LigRec.DB$target %>% unique()
  recs_in_db <- intersect(recs_in_db, rownames(ser_obj))

  # clusters <- ser_obj@active.ident %>% as.character() %>% unique()
  # df_markers_ligs <- lapply(clusters, function(cluster){
  #
  #   df <- FindMarkers(ser_obj, ident.1 = cluster, features = ligs_in_db, only.pos = T,
  #                     logfc.threshold = 0.1,min.pct = min.pct)
  #   df$gene <- rownames(df)
  #   df$ident.1 <- cluster
  #   df
  #
  # }) %>% do.call('rbind',.)
  #
  # Ligs_up_list <- split(df_markers_ligs$gene,df_markers_ligs$ident.1)
  # str(Ligs_up_list)

  # Check available assays to debug the issue
  available_assays <- names(ser_obj@assays)
  print("Available Assays: ")
  print(available_assays)
  
  # Selecting the correct assay
  if ("SCT" %in% available_assays) {
    data <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
  } else if ('RNA' %in% available_assays) {  
    data <- as.matrix(GetAssayData(ser_obj, "data", "RNA"))
  } else {
    # If neither SCT nor RNA is available, use the "spatial" assay
    data <- as.matrix(GetAssayData(ser_obj, "data", "Spatial"))
  }
  
  BarCluTable <- data.frame(Barcode = rownames(ser_obj@meta.data),
                            Cluster = ser_obj@meta.data$Cluster)

  clusters <- BarCluTable$Cluster %>% as.character() %>% unique()

  meanExpr_of_LR <- lapply(clusters, function(cluster){

    cluster.ids <- BarCluTable$Barcode[BarCluTable$Cluster == cluster]
    source_mean <- rowMeans(data[,cluster.ids])
    names(source_mean) <- rownames(data)
    source_mean

  }) %>% do.call('cbind',.) %>% as.data.frame()
  colnames(meanExpr_of_LR) <- clusters

  pct_of_LR <- lapply(clusters, function(cluster){

    cluster.ids <- BarCluTable$Barcode[BarCluTable$Cluster == cluster]
    dat <- data[,cluster.ids]
    pct <- rowSums(dat>0)/ncol(dat)
    names(pct) <- rownames(data)
    pct

  }) %>% do.call('cbind',.) %>% as.data.frame()
  colnames(pct_of_LR) <- clusters

  # calculate receptor list
  Recs_expr_list <- lapply(clusters, function(cluster){

    recs <- rownames(data)[meanExpr_of_LR[,cluster] >= expr.ct & pct_of_LR[,cluster] >= pct.ct]
    intersect(recs, recs_in_db)

  })
  names(Recs_expr_list) <- clusters
  str(Recs_expr_list)

  # calculate ligand list
  Ligs_expr_list <- lapply(clusters, function(cluster){

    ligs <- rownames(data)[meanExpr_of_LR[,cluster] >= expr.ct & pct_of_LR[,cluster] >= pct.ct]
    intersect(ligs, ligs_in_db)

  })
  names(Ligs_expr_list) <- clusters
  str(Ligs_expr_list)

  LRTG_list <- list(TGs_list = ICGs_list, Ligs_expr_list = Ligs_expr_list,Recs_expr_list = Recs_expr_list)
  return(LRTG_list)
}
