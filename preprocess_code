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


  ligs_in_db <- Databases$LigRecDB$source %>% unique()
  ligs_in_db <- intersect(ligs_in_db, rownames(ser_obj))
  recs_in_db <- Databases$LigRecDB$target %>% unique()
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

save_py_result <- function(ser_obj, exprMat.Impute, loca, mulNetList,TFLR_all_score, save_path){
  
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
  
  
  exprMat.Impute = as.matrix(exprMat.Impute)
  rownames(exprMat.Impute) = rownames(df_count)
  df_count = df_count[rownames(exprMat.Impute),]
  df_count = t(df_count)
  exprMat.Impute = t(exprMat.Impute)

  gene_id = rownames(ser_obj)
  annoMat = data.frame(Barcode=colnames(ser_obj),Cluster=ser_obj$Cluster)

  mulNet_tab <- lapply(mulNetList, function(mlnet){

    ligrec <- data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
    rectf <- data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
    tftg <- data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)

    res <- ligrec %>% merge(., rectf, by = 'Receptor') %>%
      merge(., tftg, by = 'TF') %>%
      dplyr::select(Ligand, Receptor, TF, Target) %>%
      arrange(Ligand, Receptor)

  })
  mulNet_tab <- do.call("rbind", mulNet_tab)

  # LRpairs <- unique(paste0(mulNet_tab$Ligand,"_",mulNet_tab$Receptor
  LR_link <- data.frame(ligand = mulNet_tab$Ligand, receptor = mulNet_tab$Receptor)
  TFTG_link <- data.frame(TF = mulNet_tab$TF, TG = mulNet_tab$Target)
  # TFTG_link$TF <- gsub("-", ".", TFTG_link$TF)
  # TFTG_link$TG <- gsub("-", ".", TFTG_link$TG)

  wd_score <- paste0(save_path,"TFLR_score/")
  dir.create(wd_score,recursive = T)

  TFLR_allscore = TFLR_all_score$TFLR_allscore
  cell_id = names(TFLR_allscore)

  for (id in cell_id){

    x = TFLR_allscore[[id]] %>% as.data.frame(.)
    write.table(x, file=paste0(wd_score, id, '_TFLR_score.csv'), append= T, sep=','
                ,row.names = FALSE,col.names = TRUE)

  }

  write.table(df_count,file=paste0(output_fpath, 'raw_expression_mtx.csv'),
              sep = ",",row.names = TRUE,col.names = TRUE)

  write.table(exprMat.Impute,file=paste0(output_fpath, 'imputation_expression_mtx.csv'),
              sep = ",",row.names = TRUE,col.names = TRUE)

  write.table(annoMat,file=paste0(output_fpath, 'cell_meta.csv'),
              sep = ",",row.names = TRUE,col.names = TRUE)

  write.table(loca,file=paste0(output_fpath, 'cell_location.csv'),
              sep = ",",row.names = TRUE,col.names = TRUE)

  write.table(LR_link,file=paste0(output_fpath, 'LR_links.csv'),
              sep = ",",row.names = FALSE,col.names = TRUE)

  write.table(TFTG_link,file=paste0(output_fpath, 'TFTG_links.csv'),
              sep = ",",row.names = FALSE,col.names = TRUE)

}


save_py_result_v2 <- function(mulNetList,TFLR_all_score, save_path){
  
  mulNet_tab <- lapply(mulNetList, function(mlnet){
    
    ligrec <- data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
    rectf <- data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
    tftg <- data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)
    
    res <- ligrec %>% merge(., rectf, by = 'Receptor') %>%
      merge(., tftg, by = 'TF') %>%
      dplyr::select(Ligand, Receptor, TF, Target) %>%
      arrange(Ligand, Receptor)
    
  })
  mulNet_tab <- do.call("rbind", mulNet_tab)
  
  # LRpairs <- unique(paste0(mulNet_tab$Ligand,"_",mulNet_tab$Receptor
  LR_link <- data.frame(ligand = mulNet_tab$Ligand, receptor = mulNet_tab$Receptor)
  TFTG_link <- data.frame(TF = mulNet_tab$TF, TG = mulNet_tab$Target)
  # TFTG_link$TF <- gsub("-", ".", TFTG_link$TF)
  # TFTG_link$TG <- gsub("-", ".", TFTG_link$TG)
  
  wd_score <- paste0(save_path,"TFLR_score/")
  dir.create(wd_score,recursive = T)
  
  TFLR_allscore = TFLR_all_score$TFLR_allscore
  cell_id = names(TFLR_allscore)
  
  for (id in cell_id){
    
    x = TFLR_allscore[[id]] %>% as.data.frame(.)
    write.table(x, file=paste0(wd_score, id, '_TFLR_score.csv'), append= T, sep=','
                ,row.names = FALSE,col.names = TRUE)
    
  }
  
  write.table(LR_link,file=paste0(output_fpath, 'LR_links.csv'),
              sep = ",",row.names = FALSE,col.names = TRUE)
  
  write.table(TFTG_link,file=paste0(output_fpath, 'TFTG_links.csv'),
              sep = ",",row.names = FALSE,col.names = TRUE)
  
}



select_TG <- function(count, df_loca, df_anno, python_path, Databases, min_cells, min_pvalue){

  my_python_path = python_path

  myinstr = createGiottoInstructions(python_path = my_python_path)
  gio_obj <- createGiottoObject(raw_exprs = count,
                                spatial_locs = df_loca,
                                instructions = myinstr)

  gio_obj <- addCellMetadata(gobject = gio_obj,
                             new_metadata = df_anno$Cluster,
                             vector_name = "celltype")
  # normalize
  gio_obj <- normalizeGiotto(gio_obj)
  gio_obj <- addStatistics(gobject = gio_obj)
  gio_obj <- createSpatialNetwork(gobject = gio_obj, method = "Delaunay")

  # select top 25th highest expressing genes
  gene_metadata <- fDataDT(gio_obj)
  high_expressed_genes <- gene_metadata[mean_expr_det > 0.75]$gene_ID  # mean_expr_det > 0.75

  # identify ICGs
  CPGscoresHighGenes =  findICG(gobject = gio_obj,
                                selected_genes = high_expressed_genes,
                                spatial_network_name = 'Delaunay_network',
                                cluster_column = 'celltype',
                                diff_test = 'permutation',
                                offset = 0.1,
                                adjust_method = 'fdr',
                                nr_permutations = 500,
                                do_parallel = T, cores = 6)
  # filter ICGs
  CPGscoresFilt = filterICG(CPGscoresHighGenes, direction = "both",
                            min_cells = min_cells,
                            min_cells_expr = 0.5,
                            min_int_cells = 2,
                            min_int_cells_expr = 0.5,
                            min_fdr = min_pvalue,
                            min_spat_diff = 0.2,
                            min_log2_fc = 0.2,
                            min_zscore = 1)

  ICGs_list = lapply(unique(CPGscoresFilt$CPGscores$cell_type), function(x){
    y=CPGscoresFilt$CPGscores[CPGscoresFilt$CPGscores$cell_type==x,]
    z=lapply(unique(y$int_cell_type), function(t){
      y$genes[y$int_cell_type==t]
    })
    names(z)=unique(y$int_cell_type)
    z
  })
  names(ICGs_list) = unique(CPGscoresFilt$CPGscores$cell_type)
  str(ICGs_list)

  return(ICGs_list)
}

select_LR <- function(ser_obj, Databases, pct.ct, expr.ct){

  ligs_in_db <- Databases$LigRecDB$source %>% unique()
  ligs_in_db <- intersect(ligs_in_db, rownames(ser_obj))
  recs_in_db <- Databases$LigRecDB$target %>% unique()
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

  data <- as.matrix(GetAssayData(ser_obj, "data", "SCT"))
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

  LigRec_list <- list(Ligs_expr_list = Ligs_expr_list,Recs_expr_list = Recs_expr_list)
  return(LigRec_list)
}



run_Imputation <- function(exprMat, use.seed = TRUE, seed = 2021)
{

  expr.Impute <- CreateSeuratObject(exprMat,verbose=F)
  if(use.seed) set.seed(seed)
  message('Using imputation method ALRA wrapped in Seurat')
  expr.Impute <- RunALRA(expr.Impute)
  exprMat.Impute <- expr.Impute@assays$alra@data

  return(exprMat.Impute)

}

calculate_LRTF_allscore <- function(exprMat, distMat, annoMat, group = NULL,
                                    mulNetList, Receiver, Sender = NULL,
                                    far.ct = 0.75, close.ct = 0.25,
                                    downsample = FALSE){

  if(is.null(Sender)){

    mulNetList = mulNetList[grep(paste0("-",Receiver),names(mulNetList))]
    mulNet_tab = lapply(mulNetList, function(mlnet){

      ligrec = data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
      rectf = data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
      tftg = data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)

      res = ligrec %>% merge(., rectf, by = 'Receptor') %>%
        merge(., tftg, by = 'TF') %>%
        dplyr::select(Ligand, Receptor, TF, Target) %>%
        arrange(Ligand, Receptor)

    })
    mulNet_tab = do.call("rbind", mulNet_tab)
    #by():根据mulNet_tab$Target对list进行处理
    LRpairs = by(mulNet_tab, as.character(mulNet_tab$TF), function(x){paste(x$Ligand, x$Receptor, sep = "_")})
    LRpairs = lapply(LRpairs, function(lrtf){lrtf[!duplicated(lrtf)]})
    TFs = names(LRpairs)

    cat(paste0("calculate the regulatory score of LR pairs from microenvironment to ",Receiver))
    LRTF_allscore = calculate_LRTF_score(exprMat, distMat, annoMat, group,
                                          LRpairs, TFs, Receiver, Sender,
                                          far.ct, close.ct, downsample)

  }else if(length(Sender)==1){

    cellpair = paste(Sender,Receiver,sep = "-")
    mulNet = mulNetList[[cellpair]]

    TFs = mulNet %>% .[['TFTar']] %>% dplyr::select(source) %>% unlist() %>% as.character() %>% unique()
    LRpairs = get_LRTF_link(mulNet, TFs)

    cat(paste0("calculate the regulatory score of LR pairs from ",Sender,' to ',Receiver))
    LRTF_allscore = calculate_LRTF_score(exprMat, distMat, annoMat, group,
                                         LRpairs, TFs, Receiver, Sender,
                                         far.ct, close.ct, downsample)

  }else{

    cellpair = paste(Sender,Receiver,sep = "-")
    cellpair <- intersect(names(mulNetList),cellpair)
    if(length(cellpair)==0){
      LRTF_allscore = NA
      return(LRTF_allscore)
    }

    mulNetlist = mulNetList[cellpair]
    mulNet_tab = lapply(mulNetlist, function(mlnet){

      ligrec = data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
      rectf = data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
      tftg = data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)

      res = ligrec %>% merge(., rectf, by = 'Receptor') %>%
        merge(., tftg, by = 'TF') %>%
        dplyr::select(Ligand, Receptor, TF, Target) %>%
        arrange(Ligand, Receptor)

    })

    LRpairs_TFs_list <- lapply(mulNet_tab, function(ml_tab){

      lrpairs = by(ml_tab, as.character(ml_tab$source), function(x){
        paste(x$Ligand, x$Receptor, sep = "_")
      })
      lrpairs = lapply(lrpairs, function(lrtf){lrtf[!duplicated(lrtf)]})
      tfs = names(lrpairs)

      list(LRpairs = lrpairs, TFs = tfs)

    })
    names(LRpairs_TFs_list) <- names(mulNet_tab)

    LRTF_allscore <- list()
    for (cp in cellpair) {

      receiver <- gsub('.*-','',cp)
      sender <- gsub('-.*','',cp)

      LRpairs <- LRpairs_TFs_list[[cp]]$LRpairs
      TFs <- LRpairs_TFs_list[[cp]]$TFs

      cat(paste0("calculate the regulatory score of LR pairs from ",sender,' to ',receiver))
      LRTF_allscore[[cp]] = calculate_LRTF_score(exprMat, distMat, annoMat, group,
                                                 LRpairs, TFs, receiver, sender,
                                                 far.ct, close.ct, downsample)

    }
  }

  return(LRTF_allscore)
}

# The distance weights are the reciprocal function
calculate_LRTF_score <- function(exprMat, distMat, annoMat, group = NULL,
                                 LRpairs, TFs, Receiver, Sender = NULL,
                                 far.ct = 0.75, close.ct = 0.25,
                                 downsample = FALSE)
{

  receBars = annoMat %>% dplyr::filter(Cluster == Receiver) %>%
    dplyr::select(Barcode) %>% unlist() %>% as.character()
  
  sendBars = colnames(exprMat)
  
  # if(is.character(Sender)){
  #   sendBars = annoMat %>% dplyr::filter(Cluster == Sender) %>%
  #     dplyr::select(Barcode) %>% unlist() %>% as.character()
  # }else{
  #   sendBars = annoMat %>% dplyr::filter(Cluster != Receiver) %>%
  #     dplyr::select(Barcode) %>% unlist() %>% as.character()
  # }

  Receptors = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,2]})
  Ligands = lapply(LRpairs, function(lr){stringr::str_split(lr,"_",simplify = T)[,1]})

  # get exprMat of Ligand
  LigMats = lapply(TFs, function(tf){   #TG：筛选出的LRpairs对应的Target gene
    # print(tg)
    ligands = Ligands[[tf]]
    if(length(ligands)==1){
      lig_count = exprMat[ligands, sendBars]
      lig_count = matrix(lig_count,nrow = 1)
    }else{
      lig_count = exprMat[ligands, sendBars] %>% as.matrix()
    }
    rownames(lig_count) = LRpairs[[tf]]
    colnames(lig_count) = sendBars
    lig_count
  })
  names(LigMats) = TFs

  # get exprMat of Receptor
  RecMats = lapply(TFs, function(tf){
    receptors = Receptors[[tf]]
    if(length(receptors)==1){
      rec_count = exprMat[receptors, receBars]
      rec_count = matrix(rec_count,nrow = 1)
    }else{
      rec_count = exprMat[receptors, receBars] %>% as.matrix()
    }
    rownames(rec_count) = LRpairs[[tf]]
    colnames(rec_count) = receBars
    rec_count
  })
  names(RecMats) = TFs

  distMat = distMat[sendBars, receBars]

  if(!is.null(group)){
    cpMat <- get_cell_pairs(group, distMat, far.ct, close.ct)
  }else{
    cpMat <- NULL
  }

  distMat = 1/distMat

  t1 <- Sys.time(); message(paste0('Start at: ',as.character(t1)))
  LRs_score = lapply(TFs, function(tf){

    # print(tg)
    LigMat = LigMats[[tf]]
    RecMat = RecMats[[tf]]
    lr = LRpairs[[tf]]

    if(is.null(cpMat)){

      LR_score = RecMat*(LigMat%*%distMat)
      LR_score = t(LR_score) #Receptor cells * LR pairs
      colnames(LR_score) = lr
      rownames(LR_score) = receBars

    }else{

      LR_score = lapply(unique(cpMat$Receiver), function(j){
        # j = unique(cpMat$Receiver)[1]
        is <- cpMat$Sender[cpMat$Receiver == j] %>% unique()
        if(length(is)==1){
          RecMat[,j]*(LigMat[,is]*distMat[is,j])
        }else{
          RecMat[,j]*(LigMat[,is]%*%distMat[is,j])
        }
      }) %>% do.call('cbind',.) %>% t()
      colnames(LR_score) = lr
      rownames(LR_score) = unique(cpMat$Receiver)

    }
    LR_score

  })
  names(LRs_score) = TFs
  t2 <- Sys.time(); message(paste0('End at: ',as.character(t2)))
  t2-t1

  if(is.null(cpMat)){

    TFs_expr = exprMat[,receBars]
    TFs_expr = lapply(TFs, function(tf){ exprMat[tf, receBars] })

  }else{

    TFs_expr = lapply(TFs, function(tf){ exprMat[tf, unique(cpMat$Receiver)] })

  }
  names(TFs_expr) = TFs

  # downsample
  if(length(receBars)>500){
    if(isTRUE(downsample)){
      set.seed(2021)

      if(is.null(cpMat)){
        keep_cell = sample(receBars, size = 500, replace = F)
      }else{
        keep_cell = sample(unique(cpMat$Receiver), size = 500, replace = F)
      }

      LRs_score = lapply(LRs_score, function(LR_score){ LR_score = LR_score[keep_cell,] })
      TFs_expr = lapply(TFs_expr, function(TF_count){ TF_count = TF_count[keep_cell] })
    }
  }

  LRTF_score = list(LRs_score = LRs_score, TFs_expr = TFs_expr)

  return(LRTF_score)

}


get_LRTF_link <- function(mulNet, TFs)
{

  ligrec = data.frame(Ligand = mulNet$LigRec$source, Receptor = mulNet$LigRec$target)
  rectf = data.frame(Receptor = mulNet$RecTF$source, TF = mulNet$RecTF$target)
  tftg = data.frame(TF = mulNet$TFTar$source, Target = mulNet$TFTar$target)

  mulNet_tab = ligrec %>% merge(., rectf, by = 'Receptor') %>%
    merge(., tftg, by = 'TF') %>%
    dplyr::select(Ligand, Receptor, TF, Target) %>%
    arrange(Ligand, Receptor) %>%
    filter(TF %in% TFs)

  LRTF_link = by(mulNet_tab, as.character(mulNet_tab$TF), function(x){paste(x$Ligand, x$Receptor, sep = "_")})
  LRTF_link = lapply(LRTF_link, function(lrtf){lrtf[!duplicated(lrtf)]})
  LRTF_link = LRTF_link[TFs]

  return(LRTF_link)
}

get_TFLR_allactivity <- function(exprMat = exprMat.Impute, mulNetList,LRTF_score_files, wd_model){
 
  # load TFLR score

  TFLR_allscore <- list()
  for (f in LRTF_score_files){

    message(f)

    cellpair <- gsub("LRTF_allscore_|.rds","",f)

    Receiver <- gsub('.*-','',cellpair)
    Sender <- gsub('-.*','',cellpair)

    LRTF_allscore <- readRDS(paste0(wd_model,f))
    
    if (Sender == "TME"){
      
      mulNet = mulNetList[grep(paste0("-",Receiver),names(mulNetList))]
      mulNet_tab = lapply(mulNet, function(mlnet){
        
        ligrec = data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
        rectf = data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
        tftg = data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)
        
        res = ligrec %>% merge(., rectf, by = 'Receptor') %>%
          merge(., tftg, by = 'TF') %>%
          dplyr::select(Ligand, Receptor, TF, Target) %>%
          arrange(Ligand, Receptor)
        
      })
      mulNet_tab = do.call("rbind", mulNet_tab)
      TFLR_score <- get_TFLR_activity(mulNet_tab,LRTF_allscore)
      
    }else if(length(Sender)==1){
      
      cellpair = paste(Sender,Receiver,sep = "-")
      mulNet = mulNetList[[cellpair]]
      ligrec = data.frame(Ligand = mulNet$LigRec$source, Receptor = mulNet$LigRec$target)
      rectf = data.frame(Receptor = mulNet$RecTF$source, TF = mulNet$RecTF$target)
      tftg = data.frame(TF = mulNet$TFTar$source, Target = mulNet$TFTar$target)  
      mulNet_tab = ligrec %>% merge(., rectf, by = 'Receptor') %>%
        merge(., tftg, by = 'TF') %>%
        dplyr::select(Ligand, Receptor, TF, Target) %>%
        arrange(Ligand, Receptor)
      
      TFLR_score <- get_TFLR_activity(mulNet_tab,LRTF_allscore)
      
    }
    

    if(length(LRTF_allscore$LRs_score)!=0){
      saveRDS(TFLR_score, paste0(wd_model,"TFLR_allscore_",Sender,'_',Receiver,'.rds'))
    }

    TFLR_allscore = append(TFLR_allscore, TFLR_score)
  }

  
  mulNet_alltab = lapply(mulNetList, function(mlnet){
    
    ligrec = data.frame(Ligand = mlnet$LigRec$source, Receptor = mlnet$LigRec$target)
    rectf = data.frame(Receptor = mlnet$RecTF$source, TF = mlnet$RecTF$target)
    tftg = data.frame(TF = mlnet$TFTar$source, Target = mlnet$TFTar$target)
    
    res = ligrec %>% merge(., rectf, by = 'Receptor') %>%
      merge(., tftg, by = 'TF') %>%
      dplyr::select(Ligand, Receptor, TF, Target) %>%
      arrange(Ligand, Receptor)
    
  })
  mulNet_alltab = do.call("rbind", mulNet_alltab)
  
  TFs = unique(mulNet_alltab$TF)
  TGs = unique(mulNet_alltab$Target)
  LRpairs = unique(paste0(mulNet_alltab$Ligand,"_", mulNet_alltab$Receptor))
  cell_id = names(TFLR_allscore)

  TFLR_allscore_new <- list()
  for (i in cell_id){

    tflr_score <- matrix(data=0, nrow = length(TFs), ncol = length(LRpairs))
    rownames(tflr_score) <- TFs
    colnames(tflr_score) <- LRpairs

    LR_score <- TFLR_allscore[[i]]
    tflr_score[rownames(LR_score), colnames(LR_score)] = LR_score

    TFLR_allscore_new[[i]] = tflr_score
  }
  names(TFLR_allscore_new) = cell_id

  #TFs_all_expr = exprMat[TFs, ]
  #TGs_all_expr = exprMat[TGs, ]

  # get TFTG

  TFTG_link = data.frame(TF = mulNet_tab$TF, TG = mulNet_tab$Target)

  # TFTG_link = by(mulNet_tab, as.character(mulNet_tab$Target), function(x){x$TF})
  # TFTG_link = lapply(TFTG_link, function(tftg){tftg[!duplicated(tftg)]})
  # TFTG_link = TFTG_link[TGs]

  TFLR_all = list(TFLR_allscore = TFLR_allscore_new, LRpairs = LRpairs,TFTG_link = TFTG_link)

  return(TFLR_all)
}

get_TFLR_activity <- function(mulNet_tab,LRTF_allscore){

  LRpairs <- unique(paste(mulNet_tab$Ligand, mulNet_tab$Receptor, sep = "_"))
  num_LR <- length(LRpairs)
  num_TF <- length(unique(mulNet_tab$TF))

  LRs_score <- LRTF_allscore[["LRs_score"]]
  TFs_exprMat <- LRTF_allscore[["TFs_expr"]] %>% as.data.frame(.)

  TFs <- names(LRs_score)
  cell_id <- rownames(LRs_score[[1]])

  TFLR_score <- list()
  for (i in 1:length(cell_id)){

    tflr_score <- matrix(data = 0, nrow = num_TF, ncol = num_LR)
    rownames(tflr_score) <- TFs
    colnames(tflr_score) <- LRpairs

    for (tf in TFs){

      LR_score = LRs_score[[tf]]
      cell_score = LR_score[i,]
      ind = which(colnames(tflr_score) %in% colnames(LR_score))
      tflr_score[tf,ind] = cell_score

    }
    TFLR_score[[i]] = tflr_score
  }
  names(TFLR_score) = cell_id

  # TFs_expr = lapply(cell_id, function(cell){TFs_exprMat[cell,] })
  # names(TFs_expr) = cell_id
  #
  # TFLR_allscore <- list(TFLR_score = TFLR_score, TFs_expr = TFs_expr )

  return(TFLR_score)
}



