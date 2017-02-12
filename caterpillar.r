# Kaggle competition caterpillar-tube-pricing

options(warn = 1)

library (data.table)
library (cluster)
library (xgboost)
library (randomForest)
library (Matrix)
library (ComputeBackend)

source ('stacking-algorithms.r') # TBP separately

train.mode = 'single' # { single, tune.single, bag.single, stack }
debug.mode = F

do.preprocess   = T
do.cv           = T
do.sanity.check = F

# Which stacking stages to run
do.generate.l1.train.data = T
do.generate.l1.test.data  = T
do.generate.l2.train.data = T
do.generate.l2.test.data  = T
do.generate.l3.train.data = T
do.generate.l3.test.data  = T
do.generate.submission    = T

config = list()
config$compute.backend = 'multicore' # {serial, multicore, condor, pbs}
config$nr.cores = ifelse(config$compute.backend %in% c('condor', 'pbs'), 500, 8)
config$rng.seed = 1986
config$use.irp = F

config$package.dependencies = c('ComputeBackend', 'nnls', 'R.matlab', 'kknn', 'quadprog', 'xgboost', 'Matrix', 'methods', 'liso', 'randomForest')
config$source.dependencies  = c('stacking-algorithms.r', 'isotonic-regression.r', 'cross-validation.r')
config$cluster.dependencies = c('stacking-algorithms.r', 'isotonic-regression.r', 'cross-validation.r')

config$nr.cv.folds = 5
config$nr.L0.folds = 8 # TODO probalby need more, since this affects the final model
config$nr.L1.folds = 8 # ditto
config$nr.L2.folds = 8 # ditto

# Helper function - merge rare factor levels according to the Huffman coding algorithm
merge.levels = function(x, max.levels = 32, min.freq = 100) {
  freq = data.frame(table(x))
  freq = freq[order(freq$Freq), ]
  new.levels = data.frame(old = freq$x, new = freq$x)
  
  while (freq$Freq[1] < min.freq || nrow(freq) > max.levels) {
    new.levels[new.levels$new == freq$x[1], 2] = freq$x[2]
    freq$Freq[2] = freq$Freq[1] + freq$Freq[2]
    freq = freq[-1, ]
    freq = freq[order(freq$Freq), ]
  }
  
  return (new.levels)
}

cluster.components = function(indir) {
  datadir = paste0(indir, '/competition_data')
  
  # We are clustering each type separately since they seem too different
  # We'll return a table with one cluster per component id
  comp.clust = NULL
  
  if (1) {
    # Let's start with a very naive implementation that doesn't bother with the
    # particulatiries of each table
    
    comp.files = dir(datadir)[grep('comp_', dir(datadir))]
    
    for (f in comp.files) {
      comp.type = substr(f, 6, nchar(f) - 4)
      d = read.csv(paste0(datadir, '/', f), sep = ',', quote = '')
      for (i in 1:ncol(d)) {
        if (is.numeric(d[[i]])) {
          d[[i]][d[[i]] == 9999] = NA # nasty...
        }
      }
      
      # FIXME need to spend more quality time with this to do it effectively...
      # In particular, I'm not sure what clara does with NAs and categoricals
      fit = clara(d, min(4, ceiling(nrow(d) / 5)))
      
      comp.clust = rbind(comp.clust, cbind(as.character(d$component_id), paste(comp.type, fit$clustering,sep =  '_')))
    }
  }
  else
  {
    # Bother with the particularities...
    
    #adaptors  = read.csv(paste0(datadir, '/comp_adaptor.csv' ), header = T)
    #bosses    = read.csv(paste0(datadir, '/comp_boss.csv'    ), header = T)
    #elbows    = read.csv(paste0(datadir, '/comp_elbow.csv'   ), header = T)
    #floats    = read.csv(paste0(datadir, '/comp_float.csv'   ), header = T)
    #hfls      = read.csv(paste0(datadir, '/comp_hfl.csv'     ), header = T)
    #nuts      = read.csv(paste0(datadir, '/comp_nut.csv'     ), header = T)
    #others    = read.csv(paste0(datadir, '/comp_other.csv'   ), header = T)
    #sleeves   = read.csv(paste0(datadir, '/comp_sleeve.csv'  ), header = T)
    #straights = read.csv(paste0(datadir, '/comp_straight.csv'), header = T)
    #tees      = read.csv(paste0(datadir, '/comp_tee.csv'     ), header = T)
    #threadeds = read.csv(paste0(datadir, '/comp_threaded.csv'), header = T)
    
    # adaptors
    # component_id, component_type_id, 
    # adaptor_angle, overall_length, hex_size, unique_feature, orientation, weight
    # end_form_id_1, connection_type_id_1, length_1, thread_size_1, thread_pitch_1, nominal_size_1,
    # end_form_id_2, connection_type_id_2, length_2, thread_size_2, thread_pitch_2, nominal_size_2, 
    
    # bosses
    # component_id, component_type_id, 
    # type, connection_type_id, outside_shape, 
    # base_type, height_over_tube, bolt_pattern_long, bolt_pattern_wide, 
    # groove, base_diameter, shoulder_diameter, unique_feature, orientation, weight
    
    # elbows
    # component_id, component_type_id, 
    # bolt_pattern_long, bolt_pattern_wide, 
    # extension_length, overall_length, thickness, drop_length, elbow_angle, 
    # mj_class_code, mj_plug_class_code, plug_diameter, groove, unique_feature, 
    # orientation, weight
    
    # floats
    # component_id, component_type_id, 
    # bolt_pattern_long, bolt_pattern_wide, 
    # thickness, orientation, weight
    
    # hfls
    # component_id, component_type_id, 
    # hose_diameter, corresponding_shell, 
    # coupling_class, material, plating, orientation, weight
    
    # nuts
    # component_id, component_type_id, 
    # hex_nut_size, seat_angle, 
    # length, thread_size, thread_pitch, diameter, blind_hole, orientation, weight
    
    # others
    # component_id, [NOTE: no component_type_id; I guess it's fixed by design here]
    # part_name, weight
    
    # sleeves
    # component_id, component_type_id, 
    # connection_type_id, length, 
    # intended_nut_thread, intended_nut_pitch, unique_feature, plating, 
    # orientation, weight
    
    # straight
    # component_id, component_type_id, 
    # bolt_pattern_long, bolt_pattern_wide, 
    # head_diameter, overall_length, thickness, mj_class_code, groove, 
    # unique_feature, orientation, weight
    
    # tee
    # component_id, component_type_id, 
    # bolt_pattern_long, bolt_pattern_wide, 
    # extension_length, overall_length, thickness, drop_length, mj_class_code,
    # mj_plug_class_code, groove, unique_feature, orientation, weight
    
    # threadeds
    # component_id, component_type_id, 
    # adaptor_angle, overall_length, hex_size, 
    # end_form_id_1, connection_type_id_1, length_1, thread_size_1, 
    # thread_pitch_1, nominal_size_1, end_form_id_2, connection_type_id_2, 
    # length_2, thread_size_2, thread_pitch_2, nominal_size_2, end_form_id_3, 
    # connection_type_id_3, length_3, thread_size_3, thread_pitch_3, 
    # nominal_size_3, end_form_id_4, connection_type_id_4, length_4,
    # thread_size_4, thread_pitch_4, nominal_size_4, unique_feature, orientation,
    # weight
  }
  
  comp.clust = data.frame(comp.clust)
  names(comp.clust) = c('component_id', 'component_cluster')
  
  return (comp.clust)
}

config$preprocess = function(indir, outfile, limit.levels = F) {
  #
  # Load raw data
  #
  
  datadir = paste0(indir, '/competition_data')
  
  train <- read.csv(paste0(datadir, "/train_set.csv"        ), header = T)
  test  <- read.csv(paste0(datadir, "/test_set.csv"         ), header = T)
  bom   <- read.csv(paste0(datadir, "/bill_of_materials.csv"), header = T)
  tube  <- read.csv(paste0(datadir, "/tube.csv"             ), header = T)
  specs <- read.csv(paste0(datadir, "/specs.csv"            ), header = T)
  endf  <- read.csv(paste0(datadir, "/tube_end_form.csv"    ), header = T)
  
  comps = cluster.components(indir)
  
  # Let's make sure there is no new leakage introduced unintentionally
  train.ids = -(1:nrow(train)) # these will match label/target/response values
  train.costs = train$cost
  train$id = train.ids
  train$cost = NULL
  
  #
  # Merge tables
  #
  
  data  <- rbind(train, test)
  
  data  <- merge(data, tube, by = "tube_assembly_id", all.x = T)
  data  <- merge(data, bom , by = "tube_assembly_id", all.x = T)
  
  # It's unclear how to use the components and specs features efficiently 
  # because they are unordered sets of features.
  
  if (0) {
    # Taken originally from
    # https://www.kaggle.com/ademyttenaere/caterpillar-tube-pricing/build-complete-train-and-test-db
    
    # Merge specs
    data = merge(data, specs, by = "tube_assembly_id", all.x = T)
    
    # Merge component data
    compFiles = dir(datadir)[grep("comp_", dir(datadir))]
    
    for (idComp in 1:8) {
      cat("Merging component features", idComp, "of 8\n")
      
      #d = comps
      #names(d) = paste0(names(d), '_', idComp)
      #data = merge(data, d, by = paste0("component_id_", idComp), all.x = T)
      
      for (f in compFiles) {
        compType = substr(f, 6, nchar(f) - 4)
        d = read.csv(paste0(datadir, '/', f), sep = ',', quote = "")
        names(d) = paste0(compType, '_', idComp, '_', names(d))
        data = merge(data, d, by.x = paste0("component_id_", idComp), by.y = paste0(compType, '_', idComp, '_', 'component_id'), all.x = T)
      }
    }
  } else {
    # Merge specs
    data = merge(data, specs, by = "tube_assembly_id", all.x = T)
    
    # Merge component data
    comp_details = NULL
    compFiles = dir(datadir)[grep('comp_', dir(datadir))]
    for (f in compFiles) {
      d = read.csv(paste0(datadir, '/', f), sep = ',', quote = "")
      
      if (!('unique_feature' %in% names(d))) {
        d$unique_feature = 'No'
      }
      if (!('orientation' %in% names(d))) {
        d$orientation = NA
      }
      if ('overall_length' %in% names(d)) {
        names(d)[names(d) == 'overall_length'] = 'length'
      }
      if (!('length' %in% names(d))) {
        d$length = NA
      }
      
      comp_details = rbind(comp_details, d[, c('component_id', 'weight', 'unique_feature', 'orientation', 'length')])
    }
    
    for (idComp in 1:8) {
      cat("Merging component features", idComp, "of 8\n")
      
      d = comps
      names(d) = paste0(names(d), '_', idComp)
      data = merge(data, d, by = paste0("component_id_", idComp), all.x = T)
      
      d = comp_details
      names(d) = paste0(names(d), '_', idComp)
      data = merge(data, d, by = paste0("component_id_", idComp), all.x = T)
    }
  }
  
  #
  # Some feature engineering
  #
  
  # Quote date stuff
  quote_date = strptime(data$quote_date, format = "%Y-%m-%d", tz = "GMT")
  
  data$year  = year (as.IDate(quote_date))
  data$month = month(as.IDate(quote_date))
  data$week  = week (as.IDate(quote_date))
  
  dates = data.frame(table(data$quote_date))
  names(dates) = c('quote_date', 'nr_quotes_today')
  data = merge(data, dates, by = 'quote_date', all.x = T)
  
  dates = data.frame(table(data$year))
  names(dates) = c('year', 'nr_quotes_this_year')
  data = merge(data, dates, by = 'year', all.x = T)
  
  data$month1 = data$year + data$month / 12
  data$week1 = data$year + data$week / 52
  
  dates = data.frame(table(data$month1))
  names(dates) = c('month1', 'nr_quotes_this_month')
  data = merge(data, dates, by = 'month1', all.x = T)
  
  dates = data.frame(table(data$week1))
  names(dates) = c('week1', 'nr_quotes_this_week')
  data = merge(data, dates, by = 'week1', all.x = T)
  
  data$week1 = data$month1 = NULL # (I'll delete quote_date later)
  
  # TODO there are a few dates that are shared by many examples. Can we use this somehow?
  
  # Relevant quantity
  data$qty = ifelse(data$bracket_pricing == 'Yes', data$quantity, data$min_order_quantity)
  data$clean.quantity = pmax(data$quantity, data$min_order_quantity) # ?? not sure what's going on in the 2% datapoints where this does something
  data$clean.qty = ifelse(data$bracket_pricing == 'Yes', data$clean.quantity, data$min_order_quantity)
  data$annualized_qty = pmax(data$annual_usage, data$qty) # is annual usage below an order quantity (and especially 0) actually a missing value?
  data$annualized_qty2 = data$clean.qty + 0.5 * data$annual_usage # annual quantity is just an estimate, not guaranteed
  
  # not sure if I should leave these in or drop them?
  if (0) {
    data$quantity = data$min_order_quantity = data$annual_usage = NULL
    data$clean.quantity = data$qty = NULL
  }
  
  # Quote sets (TODO: I'm sure there is much more to do here, and way beyond feature engineering)
  dt = data.table(data) # FIXME I really need to learn the advanced syntax for this... next comp!
  dt = dt[, qcount := .N            , by = list(tube_assembly_id, supplier, quote_date)]
  dt = dt[, qmax   := max(clean.qty), by = list(tube_assembly_id, supplier, quote_date)]
  dt = dt[, qmin   := min(clean.qty), by = list(tube_assembly_id, supplier, quote_date)]
  data$qs_count = dt$qcount
  data$qs_max = dt$qmax
  data$qs_range = log(dt$qmax / ifelse(dt$qmin == 0, 1, dt$qmin))
  data$qs_rel_qty = (data$clean.qty - dt$qmin) / ifelse(dt$qmax - dt$qmin == 0, 1, dt$qmax - dt$qmin)
  
  # How many nontrivial ends
  data$nr_interesting_ends = (data$end_x != 'NONE') + (data$end_a != 'NONE')
  
  # How many formed ends
  formed_end_type_ids = endf$end_form_id[endf$forming == 'Yes']
  data$nr_formed_ends = (data$end_a %in% formed_end_type_ids) + (data$end_x %in% formed_end_type_ids)
  
  # How many special tooling ends
  data$nr_special_ends = (data$end_a_1x == 'Y') + (data$end_x_1x == 'Y') + (data$end_a_2x == 'Y') + (data$end_x_2x == 'Y') # note that in all sample with *_2x == Y, so is *_1x; this still doesn't code the levels perfectly, but hopefully captures the signal well
  
  # again, not sure if I should drop these
  data$end_a_1x = data$end_a_2x = data$end_x_1x = data$end_x_2x = NULL
  
  # Dimensions
  data$volume1 = data$diameter * data$length * data$wall # material volume
  data$volume2 = data$diameter ^ 2 * data$length # ~ storage volume if wall inclusive
  data$volume3 = (data$diameter ^ 2 - (data$diameter - data$wall) ^ 2) * data$length # ~ storage volume if wall not inclusive
  data$bend_rx = data$bend_radius * data$num_bends # not really sure what this does..
  
  # Some models don't handle categorical variables very well; try to extract some
  # of the signal manually:
  
  # Total number of quotes obtained from supplier (in both train and test sets! use leak / transduce)
  suppliers = data.frame(table(data$supplier))
  names(suppliers) = c('supplier', 'supplier_quote_vol')
  data = merge(data, suppliers, by = 'supplier', all.x = T)
  
  # Total number of quotes from material id (in both train and test sets! use leak / transduce)
  materials = data.frame(table(data$material_id))
  names(materials) = c('material_id', 'material_quote_vol')
  data = merge(data, materials, by = 'material_id', all.x = T)
  
  # people said post-competition that additional statistics accross supplier were useful
  if (1) {
    dt = data.table(data)
    dt = dt[, qmax  := max(clean.qty)   , by = list(supplier)] # maybe limit the date distance for a cleaner grouping?
    dt = dt[, qmin  := min(clean.qty)   , by = list(supplier)]
    dt = dt[, aqmax := max(annual_usage), by = list(supplier)]
    dt = dt[, aqmin := min(annual_usage), by = list(supplier)]
    data$sup_qmin  = dt$qin
    data$sup_qmax  = dt$qmax
    data$sup_aqmin = dt$aqin
    data$sup_aqmax = dt$aqmax
    data$sup_qrel  = (data$clean.qty    - dt$qmin ) / ifelse(dt$qmax  - dt$qmin  == 0, 1, dt$qmax  - dt$qmin )
    data$sup_aqrel = (data$annual_usage - dt$aqmin) / ifelse(dt$aqmax - dt$aqmin == 0, 1, dt$aqmax - dt$aqmin)
  }
  
  data$quote_date = NULL
  
  if (0) {
    # Exploit external leakage from financial market data
    warning('Using external data (forbidden by competition rules, not for submission!)')
    
    # TODO: get the most relevant signals...
    
    # A metals prices index
    xme.prices = read.csv('~/Projects/Data/Google/Finance/xme.csv', header = T)
    xme.date = strptime(xme.prices$Date, format = "%d-%b-%y", tz = "GMT")
    xme.prices$year  = year (as.IDate(xme.date))
    xme.prices$month = month(as.IDate(xme.date))
    xme.prices$week  = week (as.IDate(xme.date))
    xme.prices$xme = as.numeric(xme.prices$Open)
    xme.prices.new = aggregate(xme ~ month + year, xme.prices, mean)
    data = merge(data, xme.prices.new, by = c('year', 'month'), all.x = T)
    
    # Caterpillar's stock
    cat.prices = read.csv('~/Projects/Data/Google/Finance/cat.csv', header = T)
    cat.date = strptime(cat.prices$Date, format = "%d-%b-%y", tz = "GMT")
    cat.prices$year  = year (as.IDate(cat.date))
    cat.prices$month = month(as.IDate(cat.date))
    cat.prices$week  = week (as.IDate(cat.date))
    cat.prices$cat = as.numeric(cat.prices$Open)
    cat.prices.new = aggregate(cat ~ month + year, cat.prices, mean)
    data = merge(data, cat.prices.new, by = c('year', 'month'), all.x = T)
  }
  
  if (1) {
    # Try to handle the unordered sets of features
    
    n.data = nrow(data)
    
    # Ends (I don't think the order is meaningful, but I could be wrong...)
    end_qty_per_type = matrix(0, n.data, length(levels(endf$end_form_id))) # number of components of each type
    idxs = seq.int(n.data) + n.data * (as.numeric(data$end_a) - 1)
    end_qty_per_type[idxs] = end_qty_per_type[idxs] + 1
    data$end_a = NULL
    idxs = seq.int(n.data) + n.data * (as.numeric(data$end_x) - 1)
    end_qty_per_type[idxs] = end_qty_per_type[idxs] + 1
    data$end_x = NULL
    data = cbind(data, end_qty_per_type = end_qty_per_type)
    
    # Components
    data$nr_comps        = 0 # number of components (taking quantity into account)
    data$nr_ucomps       = 0 # number of component types
    data$ttl_comp_weight = 0 # total component weight
    data$ttl_comp_length = 0 # total component length
    data$ttl_comp_orient = 0 # total components with orientation
    data$nr_unq_ftrs     = 0 # number of components with a "unique feature"
    
    # (people said post-competition that these were helpful)
    data$min_comp_weight = Inf
    data$max_comp_weight = -Inf
    data$min_comp_length = Inf
    data$max_comp_length = -Inf
    
    comp_qty_per_cluster = matrix(0, n.data, length(levels(comps$component_cluster))) # number of components of each cluster
    
    for (i in 1:8) {
      cid.fn = paste0('component_id_'     , i)
      ccl.fn = paste0('component_cluster_', i) # NOTE: all 8 features share the exact same set of levels
      cqt.fn = paste0('quantity_'         , i)
      cwt.fn = paste0('weight_'           , i) # I'm assuming that all values are on the same scale...
      cln.fn = paste0('length_'           , i) # I'm assuming that all values are on the same scale...
      cuf.fn = paste0('unique_feature_'   , i)
      cor.fn = paste0('orientation_'      , i)
      
      data[is.na(data[, cqt.fn]), cqt.fn] = 0
      data[is.na(data[, cwt.fn]), cwt.fn] = 0
      data[is.na(data[, cln.fn]), cln.fn] = 0
      data[is.na(data[, cuf.fn]), cuf.fn] = 'No'
      data[is.na(data[, cor.fn]), cor.fn] = 'No'
      data$nr_comps        = data$nr_comps        + data[, cqt.fn]
      data$nr_ucomps       = data$nr_ucomps       + (data[, cqt.fn] > 0)
      data$ttl_comp_weight = data$ttl_comp_weight + data[, cqt.fn] * data[, cwt.fn]
      data$ttl_comp_length = data$ttl_comp_length + data[, cqt.fn] * data[, cln.fn]
      data$ttl_comp_orient = data$ttl_comp_orient + data[, cqt.fn] * (data[, cor.fn] == 'Yes')
      data$nr_unq_ftrs     = data$nr_unq_ftrs     + data[, cqt.fn] * (data[, cuf.fn] == 'Yes')
      
      data$min_comp_weight = pmin(data$min_comp_weight, data[, cwt.fn])
      data$max_comp_weight = pmax(data$max_comp_weight, data[, cwt.fn])
      data$min_comp_length = pmin(data$min_comp_length, data[, cln.fn])
      data$max_comp_length = pmax(data$max_comp_length, data[, cln.fn])
      
      non.na.idxs = !is.na(data[, ccl.fn])
      idxs = seq.int(n.data)[non.na.idxs] + n.data * (as.numeric(data[non.na.idxs, ccl.fn]) - 1)
      comp_qty_per_cluster[idxs] = comp_qty_per_cluster[idxs] + data[non.na.idxs, cqt.fn]
      
      data = data[, -match(c(cid.fn, cqt.fn, ccl.fn, cwt.fn, cln.fn, cuf.fn, cor.fn), names(data))]
    }
    
    data = cbind(data, comp_qty_per_cluster = comp_qty_per_cluster)
    
    # Specs
    
    # These don't share the same set of levels, so let's fix that first
    all.spec.levels = NULL
    for (i in 1:10) {
      fn = paste0('spec', i) # unfortunately these don't share the same set of levels
      data[, fn] = as.character(data[, fn])
      data[is.na(data[, fn]), fn] = '<NA>'
      all.spec.levels = c(all.spec.levels, unique(data[, fn]))
    }
    all.spec.levels = unique(all.spec.levels)
    
    data$nr_specs = 0 # number of specs
    spec_per_type = matrix(0, n.data, length(all.spec.levels)) # nearly a matrix of indicators: which tube supports which spec
    
    for (i in 1:10) {
      fn = paste0('spec', i)
      data$nr_specs = data$nr_specs + (data[, fn] != '<NA>')
      data[, fn] = factor(data[, fn], levels = all.spec.levels)
      idxs = seq.int(n.data) + n.data * (as.numeric(data[, fn]) - 1)
      spec_per_type[idxs] = spec_per_type[idxs] + 1
      data = data[, -match(fn, names(data))]
    }
    
    spec_per_type = spec_per_type[, -which(all.spec.levels == '<NA>')]
    data = cbind(data, spec_per_type = spec_per_type)
  }
  
  #data$tube_assembly_id  <- NULL # this one is actually nedded for correct data splitting as in cross-validation
  
  #
  # Convert strings to factors
  #
  
  for (i in 1:ncol(data)) {
    if (is.character(data[, i])) {
      cat(names(data)[i], '\n')
      data[, i] = as.factor(data[, i])
    }
  }
  
  #
  # Get rid of missing values
  #
  
  for (i in 1:ncol(data)) {
    if (any(is.na(data[,i]))) {
      if (is.numeric(data[,i])) {
        cat('NOTE: numeric feature', names(data)[i], 'has', mean(is.na(data[,i])), 'missing values. Will impute to -999 that otherwise does not appear\n')
        #tbl = table(as.vector(data[,i]))
        #xmode = as.numeric(names(tbl)[which.max(tbl)])
        data[is.na(data[,i]),i] = 0 # currently the only missing values are quantities, so 0 makes more sense
      } else {
        cat('NOTE: categorical feature', names(data)[i], 'has', mean(is.na(data[,i])), 'missing values. Will impute to a new <NA> category\n')
        data[,i] = as.character(data[,i])
        data[is.na(data[,i]),i] = "<NA>"
        data[,i] = as.factor(data[,i])
      }
    }
  }
  
  #
  # Clean variables with too few/many categories
  #
  
  # Also note that with rare levels we might encounter them in testing when we
  # never saw them in dataing, in which case it's unclear what to predict...
  
  degenerate.list = NULL
  
  for (i in 1:ncol(data)) {
    if (names(data)[i] != 'tube_assembly_id') {
      # NOTE: with the way xgboost handles factors (i.e., indicator variables), it is not necessary to merge rare levels
      if (limit.levels && is.factor(data[, i])) {
        # FIXME huffman merging is oblivious to the response, but unclear how to use the response without leaking
        new.levels = merge.levels(data[, i])
        
        if (nrow(new.levels) != length(unique(new.levels$new))) {
          cat('NOTE: categorical feature', names(data)[i], 'has some rare levels. Will merge', nrow(new.levels) - length(unique(new.levels$new)), 'levels\n')
          levels(data[, i]) = new.levels$new # FIXME this reorders the levels, and the resulting labels don't expose any merges
        }
      }
      
      if (length(unique(data[, i])) < 2) {
        cat('NOTE: feature', names(data)[i], 'only takes one value. Will drop the feature\n')
        degenerate.list = c(degenerate.list, names(data)[i])
      }
    }
  }
  
  if (length(degenerate.list) > 0) {
    data = data[, -match(degenerate.list, names(data))]
  }
  
  #
  # Recover train and test set, clean up and save
  #
  
  train = data[data$id < 0, ]
  test  = data[data$id > 0, ]
  
  # Will need these to split the data correctly
  train.atomic.ids = train$tube_assembly_id
  
  # People say that these are still leaking a bit! (I neglected to check this 
  # during the competition because the ids were factors and I believed Kaggle 
  # would not allow Caterpillar to make this mistake....). I'm not convinced
  # though that what said people saw was actually another way of coding the 
  # rank within bracket pricing tuples (which is a feature I added directly)
  # and other features I've engineered manually
  train$tube_assembly_id = as.numeric(substr(as.character(train$tube_assembly_id), 4,8))
  test$tube_assembly_id = as.numeric(substr(as.character(test$tube_assembly_id), 4,8))
  
  train$cost[order(train$id)] = train.costs[order(train.ids)]
  
  if (0) {
    # Cluster the data for a poor man's transduction using example weights
    
    cat('Clustering all examples\n')
    fit = clara(data[, -match(c('id', 'tube_assembly_id'), names(data))], k = 60)
    data$cluster.id = fit$clustering
    
    cluster.test.weights = data.frame(table(cluster.id = fit$clustering, is.tst = data$id > 0))
    cluster.test.weights = data.frame(cluster.id = 1:60, weight = cluster.test.weights$Freq[cluster.test.weights$is.tst == 'TRUE'] / (cluster.test.weights$Freq[cluster.test.weights$is.tst == 'TRUE'] + cluster.test.weights$Freq[cluster.test.weights$is.tst == 'FALSE']))
    
    data = merge(data, cluster.test.weights, by = 'cluster.id', all.x = T)
    data = data[order(data$id), ]
    trainset.weights = data$cluster.id[data$id < 0]
    trainset.weights[order(train$id)] = trainset.weights
  } else {
    trainset.weights = rep(1, nrow(train))
  }
  
  train$id = NULL 
  test.ids = test$id
  test$id  = NULL
  
  if (0) {
    warning('Running not-for-release experiment!')
    
    # Try to figure out what determines the basal tube cost 
    # (from which volume prices are derived according to some dicsount scheme, I assume)
    # --- WIP, didn't have time to finish this...
    idxs = train$bracket_pricing == 'Yes' & train$qty == 1 & train$min_order_quantity <= 1 & train$annual_usage <= 1
    train = train[idxs, ]
    train.atomic.ids = train.atomic.ids[idxs]
    idxs = test$bracket_pricing == 'Yes' & test$qty == 1 & test$min_order_quantity <= 1 & test$annual_usage <= 1
    test = test[idxs, ]
    test.ids = test.ids[idxs]
  }
  
  save(train, test, test.ids, train.atomic.ids, trainset.weights, file = outfile)
  
  return (0)
}

config$tranform.response = function(y) {
  # NOTE: this has to match recover and feval functions below!
  return (log1p(y))
  #return (y ^ 0.25)
}

config$recover.response = function(y) {
  # NOTE: this has to match feval function below!
  return (expm1(y))
  #return (y ^ 4)
}

config$feval = function(preds, dtrain) {
  labels = getinfo(dtrain, 'label')
  
  #NOTE: this has to match transform/recover response functions above..
  
  v1 = expm1(labels)
  v2 = expm1(preds )
  #v1 = labels ^ 4
  #v2 = preds  ^ 4
  
  value = sqrt(mean((log1p(v1) - log1p(v2)) ^ 2))
  return (list(metric = 'rmsle', value = value))
}

# FIXME the stacking implementaiton here suffers from horrible code replication...

config$L0.model.names = c('xgb.rf', 'xgb.tuned', 'xgb.bag', 'rf')

config$train.L0 = function(config, train.data) {
  L0.fitted.models = list()
  
  # TODO somehow diversify the set... for now I'm assuming tuned.pars includes 
  # randomization, and so the models will be a bit different. But it's doubtful
  # that more than bagging would be needed to make the most out of them...
  #
  # => play with the xgb.pars for each model separately
  
  xgb.train.data = xgb.DMatrix(sparse.model.matrix(~ . - cost - 1, data = train.data), missing = -999, label = train.data$cost)
  
  # 1. A (simplistic?) random forest using xgboost
  xgb.pars1 = list(
    booster           = 'gbtree',
    objective         = 'reg:linear',
    verbose           = 1,
    
    eta               = 1   , # shrinkage along boosting rounds (lower will slow training convergence)
    gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 50  , # single tree constraint (effectively the maximum variable interactions per tree)
    min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.8 , # bagging-like randomization per round
    colsample_bytree  = 0.6 , # random-forest like randomization per round
    num_parallel_tree = 200   # random forest size at every stage of boosting
  )
  xgb.n.rounds1 = 1
  
  # The best xgboost I could muster
  xgb.pars2 = list(
    booster           = 'gbtree',
    objective         = 'reg:linear',
    verbose           = 1,
    
    eta               = 0.05, # shrinkage along boosting rounds (lower will slow training convergence)
    gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 6   , # single tree constraint (effectively the maximum variable interactions per tree)
    min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.8 , # bagging-like randomization per round
    colsample_bytree  = 0.6 , # random-forest like randomization per round
    num_parallel_tree = 1     # random forest size at every stage of boosting
  )
  xgb.n.rounds2 = 2000 # can squeeze more out of it with 4000 rounds
  
  # Bagging of a more aggresive xgboost
  xgb.pars3 = list(
    booster           = 'gbtree',
    objective         = 'reg:linear',
    verbose           = 1,
    
    eta               = 0.1 , # shrinkage along boosting rounds (lower will slow training convergence)
    gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 10  , # single tree constraint (effectively the maximum variable interactions per tree)
    min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.8 , # bagging-like randomization per round
    colsample_bytree  = 0.6 , # random-forest like randomization per round
    num_parallel_tree = 1     # random forest size at every stage of boosting
  )
  xgb.n.rounds3 = 200
  lin.bag.count = 10
  
  # TODO: maybe use the bagged linear booster that seemed to work pretty well in Kaggle Scripts
  
  L0.fitted.models$model1 = xgb.train(params = xgb.pars1, data = xgb.train.data, nrounds = xgb.n.rounds1, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8))
  L0.fitted.models$model2 = xgb.train(params = xgb.pars2, data = xgb.train.data, nrounds = xgb.n.rounds2, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8))
  
  L0.fitted.models$model3 = list()
  for (i in 1:lin.bag.count) {
    L0.fitted.models$model3[[i]] = xgb.train(params = xgb.pars3, data = xgb.train.data, nrounds = xgb.n.rounds3, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8))
  }
  
  L0.fitted.models$model4 = randomForest(formula = cost ~ . - supplier, data = train.data, ntree = 30)
  
  return (L0.fitted.models)
}

config$predict.L0 = function(config, L0.fitted.models, test.data) {
  xgb.test.data = xgb.DMatrix(sparse.model.matrix(~ . - 1, data = test.data), missing = -999)
  
  pred1 = predict(L0.fitted.models$model1, newdata = xgb.test.data)
  pred2 = predict(L0.fitted.models$model2, newdata = xgb.test.data)
  
  pred3 = pred1 * 0
  for (i in 1:length(L0.fitted.models$model3)) {
    pred3 = pred3 + predict(L0.fitted.models$model3[[i]], newdata = xgb.test.data)
  }
  pred3 = pred3 / length(L0.fitted.models$model3)
  
  pred4 = predict(L0.fitted.models$model4, newdata = test.data)
  
  pred = data.frame(pred1 = pred1, pred2 = pred2, pred3 = pred3, pred4 = pred4)
  return (pred)
}

config$process.L0 = function(config, core) {
  folds.this.core = compute.backend.balance(length(config$L0.folds), config$nr.cores)[[core]]
  nr.folds.this.core = length(folds.this.core)
  
  if (nr.folds.this.core < 1) {
    return (NULL)
  }
  
  L1.train.X = NULL
  L1.train.y = NULL
  L1.train.f = NULL
  
  for (fold.i.core in 1:nr.folds.this.core) {
    fold.i = folds.this.core[fold.i.core]
    fold.train.data = config$train.data[config$L0.folds[[fold.i]]$train, ]
    fold.test.data  = config$train.data[config$L0.folds[[fold.i]]$test , ]
    
    L0.fitted.models = config$train.L0(config, fold.train.data)
    L0.preds = config$predict.L0(config, L0.fitted.models, fold.test.data)
    
    L1.train.X = rbind(L1.train.X, L0.preds)
    L1.train.y = c(L1.train.y, fold.test.data$cost)
    L1.train.f = c(L1.train.f, rep(fold.i, nrow(fold.test.data)))
  }
  
  return (cbind(L1.train.f, L1.train.y, L1.train.X))
}

config$L1.model.names = c('Best', 'NNLS', 'IAM') # 'Iso', 

config$train.L1 = function(config, train.data.X, train.data.y) {
  L1.fitted.models = list()
  
  # TODO: need to implement separate predict functions for multiso and siam
  
  L1.fitted.models$best.idx  = which.min(colSums((train.data.y - train.data.X) ^ 2))
  L1.fitted.models$nnls.coef = nnls(train.data.X, train.data.y)$x
  L1.fitted.models$iam       = liso.backfit(train.data.X, train.data.y)
  #L1.fitted.models$iso.fit   = list(X = train.data.X, y = multiso.matlab.train(config$matlab, train.data.X, train.data.y))
  
  return (L1.fitted.models)
}

config$predict.L1 = function(config, L1.fitted.models, train.data.X) {
  pred1 = train.data.X[, L1.fitted.models$best.idx]
  pred2 = train.data.X %*% L1.fitted.models$nnls.coef
  pred3 = predict(L1.fitted.models$iam, train.data.X)
  #pred4 = multiso.matlab.predict(L1.fitted.models$iso.fit$X, L1.fitted.models$iso.fit$y, train.data.X)
  
  pred = data.frame(pred1 = pred1, pred2 = pred2, pred3 = pred3) # , pred4 = pred4
  return (pred)
}

config$process.L1 = function(config, core) {
  folds.this.core = compute.backend.balance(length(config$L1.folds), config$nr.cores)[[core]]
  nr.folds.this.core = length(folds.this.core)
  
  if (nr.folds.this.core < 1) {
    return (NULL)
  }
  
  if (config$use.irp) {
    old.wd = getwd()
    tmp.wd = paste0('tmp/core', core) # need each core to work in a separate dir, otherwise R.matlab gets confused
    dir.create(tmp.wd, showWarnings = F)
    file.copy('irp_wrapper.m', tmp.wd, overwrite = T)
    setwd(tmp.wd)
    matlab.port = 10000 + core
    Matlab$startServer(port = matlab.port)
    config$matlab = Matlab(port = matlab.port)
    isOpen = open(config$matlab)
  }
  
  L2.train.X = NULL
  L2.train.y = NULL
  L2.train.f = NULL
  
  for (fold.i.core in 1:nr.folds.this.core) {
    fold.i = folds.this.core[fold.i.core]
    fold.train.data.X = data.matrix(config$L1.train.data$X[config$L1.folds[[fold.i]]$train, ])
    fold.test.data.X  = data.matrix(config$L1.train.data$X[config$L1.folds[[fold.i]]$test , ])
    fold.train.data.y = config$L1.train.data$y[config$L1.folds[[fold.i]]$train]
    fold.test.data.y  = config$L1.train.data$y[config$L1.folds[[fold.i]]$test ]
    
    L1.fitted.models = config$train.L1(config, fold.train.data.X, fold.train.data.y)
    L1.preds = config$predict.L1(config, L1.fitted.models, fold.test.data.X)
    
    L2.train.X = rbind(L2.train.X, L1.preds)
    L2.train.y = c(L2.train.y, fold.test.data.y)
    L2.train.f = c(L2.train.f, rep(fold.i, nrow(fold.test.data.X)))
  }
  
  if (config$use.irp) {
    close(config$matlab)
    setwd(old.wd)
  }
  
  return (cbind(L2.train.f, L2.train.y, L2.train.X))
}

config$L2.model.names = c('Best', 'NNLS')

config$train.L2 = function(config, train.data.X, train.data.y) {
  L2.fitted.models = list()
  
  # TODO: need to implement separate predict functions for multiso and siam
  
  L2.fitted.models$best.idx  = which.min(colSums((train.data.y - train.data.X) ^ 2))
  L2.fitted.models$nnls.coef = nnls(train.data.X, train.data.y)$x
  
  return (L2.fitted.models)
}

config$predict.L2 = function(config, L2.fitted.models, train.data.X) {
  pred1 = train.data.X[, L2.fitted.models$best.idx]
  pred2 = train.data.X %*% L2.fitted.models$nnls.coef
  
  pred = data.frame(pred1 = pred1, pred2 = pred2)
  return (pred)
}

config$process.L2 = function(config, core) {
  folds.this.core = compute.backend.balance(length(config$L2.folds), config$nr.cores)[[core]]
  nr.folds.this.core = length(folds.this.core)
  
  if (nr.folds.this.core < 1) {
    return (NULL)
  }
  
  if (config$use.irp) {
    old.wd = getwd()
    tmp.wd = paste0('tmp/core', core) # need each core to work in a separate dir, otherwise R.matlab gets confused
    dir.create(tmp.wd, showWarnings = F)
    file.copy('irp_wrapper.m', tmp.wd, overwrite = T)
    setwd(tmp.wd)
    matlab.port = 10000 + core
    Matlab$startServer(port = matlab.port)
    config$matlab = Matlab(port = matlab.port)
    isOpen = open(config$matlab)
  }
  
  L3.train.X = NULL
  L3.train.y = NULL
  L3.train.f = NULL
  
  for (fold.i.core in 1:nr.folds.this.core) {
    fold.i = folds.this.core[fold.i.core]
    fold.train.data.X = data.matrix(config$L2.train.data$X[config$L2.folds[[fold.i]]$train, ])
    fold.test.data.X  = data.matrix(config$L2.train.data$X[config$L2.folds[[fold.i]]$test , ])
    fold.train.data.y = config$L2.train.data$y[config$L2.folds[[fold.i]]$train]
    fold.test.data.y  = config$L2.train.data$y[config$L2.folds[[fold.i]]$test ]
    
    L2.fitted.models = config$train.L2(config, fold.train.data.X, fold.train.data.y)
    L2.preds = config$predict.L2(config, L2.fitted.models, fold.test.data.X)
    
    L3.train.X = rbind(L3.train.X, L2.preds)
    L3.train.y = c(L3.train.y, fold.test.data.y)
    L3.train.f = c(L3.train.f, rep(fold.i, nrow(fold.test.data.X)))
  }
  
  if (config$use.irp) {
    close(config$matlab)
    setwd(old.wd)
  }
  
  return (cbind(L3.train.f, L3.train.y, L3.train.X))
}

config$process.bagged.xgb = function(config, core) {
  # Each core trains one random forest-like xgboost model, with a different seed
  
  # Apparently this structure has to be reinitialized in each R process, so:
  xgb.train.data = xgb.DMatrix(sparse.model.matrix(~ . - cost - 1, data = config$train.data), missing = -999, label = config$train.data$cost)
  xgb.test.data  = xgb.DMatrix(sparse.model.matrix(~ .        - 1, data = config$test.data ), missing = -999)
  
  # Train and predict
  model = xgb.train(params = c(config$xgb.fixed.pars, config$xgb.tuned.pars), data = xgb.train.data, nrounds = config$xgb.n.rounds, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8))
  test.preds = predict(model, xgb.test.data) # NOTE: on the transformed response scale
  
  if (!is.null(config$valid.data)) {
    xgb.valid.data = xgb.DMatrix(sparse.model.matrix(~ . - 1, data = config$valid.data), missing = -999)
    valid.preds = predict(model, xgb.valid.data) # NOTE: on the transformed response scale
    
    return (list(valid.preds = valid.preds, test.preds = test.preds))
  }
  
  return (list(test.preds = test.preds))
}

config$process.tune.xgb = function(config, core) {
  # Try some setups of tuning parameters, record cross validation error
  
  idxs.this.core = compute.backend.balance(nrow(config$xgb.tuned.pars.candidates), config$nr.cores)[[core]]
  nr.idxs.this.core = length(idxs.this.core)
  
  if (nr.idxs.this.core < 1) {
    return (NULL)
  }
  
  cv.errs = rep(NA, nr.idxs.this.core)
  
  cat(date(), 'Job', core, 'out of', config$nr.cores, 'started', '\n')
  
  # Apparently this structure has to be reinitialized in each R process, so:
  xgb.train.data = xgb.DMatrix(sparse.model.matrix(~ . - cost - 1, data = config$train.data), missing = -999, label = config$train.data$cost)
  
  for (i in 1:nr.idxs.this.core) {
    cat(date(), 'Working on tuning setup', i, 'out of', nr.idxs.this.core, '\n')
    
    xgb.tuned.pars = config$xgb.tuned.pars.candidates[idxs.this.core[i], ]
    xgb.cv.res = xgb.cv(params = c(config$xgb.fixed.pars, xgb.tuned.pars), data = xgb.train.data, nrounds = config$xgb.n.rounds, folds = config$xgb.folds, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8), early.stop.round = 100, verbose = T, print.every.n = 100, maximize = F)
    cat('\n')
    
    cv.errs[i] = min(xgb.cv.res$test.rmsle.mean + xgb.cv.res$test.rmsle.std)
  }
  
  cat(date(), 'Job', core, 'out of', config$nr.cores, 'finished', '\n')
  
  return (cv.errs)
}

# Actual work
# ==============================================================================

config$tag = paste0('afterthoughts')
config$subtag = 'Single XGB(trees) on my data'

set.seed(config$rng.seed)

if (1) {
  clean.data.file = paste0('../Results/cater-clean-data-', config$tag, '.RData')
} else {
  warning('Loading data from old preprocessing, is this a sanity check?!')
  clean.data.file = '~/Projects/Data/Kaggle/caterpillar-tube-pricing/0-24-with-xgboost-in-r-clean-data.RData'
  #clean.data.file = '~/Projects/Data/Kaggle/caterpillar-tube-pricing/0-2748-with-rf-and-log-transformation.RData'
}

if (do.preprocess) {
  stopifnot(substr(clean.data.file, 1, 10) == '../Results') # I don't want to overwrite data by mistake
  
  cat(date(), 'Preprocessing competition data\n')
  set.seed(config$rng.seed)
  config$preprocess('~/Projects/Data/Kaggle/caterpillar-tube-pricing', clean.data.file)
}

if (1) {
  cat(date(), 'Loading clean competition data\n')
  
  load(clean.data.file) # => train, test, test.ids, train.atomic.ids, test.atomic.ids
  
  if (debug.mode) {
    warning('DEBUG MODE: running on subsampled data!')
    idxs = sample(nrow(train), 6000)
    train = train[idxs, ]
    train.atomic.ids = train.atomic.ids[idxs]
  }
  
  config$train.data = train
  config$test.data = test
  config$test.ids = test.ids
  config$train.atomic.ids = train.atomic.ids
  rm(train, test, test.ids, train.atomic.ids)
  
  config$train.data$cost = config$tranform.response(config$train.data$cost)
  
  if (do.sanity.check) {
    warning('Holding out some data for sanity check')
    
    holdout.idxs = get.atomic.random.folds(config$train.atomic.ids, config$nr.cv.folds)[[1]]$test
    config$valid.data = config$train.data[holdout.idxs, ]
    config$valid.cost = config$valid.data$cost
    config$valid.data$cost = NULL
    config$train.data = config$train.data[-holdout.idxs, ]
    config$train.atomic.ids = config$train.atomic.ids[-holdout.idxs]
  }
  
  #
  # Set up CV folds
  #
  
  config$cv.folds = get.atomic.random.folds(config$train.atomic.ids, config$nr.cv.folds)
  
  #
  # Set up XGB
  #
  
  # 1. xgboost doesn't handle factors directly, so we recode them with indicators
  # 2. the value -999 never appears in the data, so it has no effect, and the 
  #    data are actually dense (which does mean it will run much more slowly, but 
  #    at least it won't treat the many zeros in the data as if they were missing!)
  
  # NOTE: this code is copied in job processing functions, so copy any change to there too!
  config$xgb.train.data = xgb.DMatrix(sparse.model.matrix(~ . - cost - 1, data = config$train.data), missing = -999, label = config$train.data$cost)
  config$xgb.test.data  = xgb.DMatrix(sparse.model.matrix(~ .        - 1, data = config$test.data ), missing = -999)
  
  if (do.sanity.check) {
    config$xgb.valid.data = xgb.DMatrix(sparse.model.matrix(~ . - 1, data = config$valid.data), missing = -999)
  }
  
  # TODO: can I somehow do a poor man's version of transductive
  # learning? (upweight examples that are similar to the specific 
  # testset we need to generate predictions for). DMatrix supports data weights
  # which are fed into xgboost.
  
  config$xgb.fixed.pars = list(
    booster          = 'gbtree',
    #booster          = 'gblinear',
    objective        = 'reg:linear',
    verbose          = 1,
    scale_pos_weight = 1
  )
  
  config$xgb.folds = lapply(config$cv.folds, function(x) { x$test })
}

if (train.mode == 'single') {
  # Train a single XGB model (useful for tuning parameters)
  
  config$xgb.tuned.pars = list(
    eta               = 0.05, # shrinkage along boosting rounds (lower will slow training convergence)
    gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 6   , # single tree constraint (effectively the maximum variable interactions per tree)
    min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.8 , # bagging-like randomization per round
    colsample_bytree  = 0.6 , # random-forest like randomization per round
    num_parallel_tree = 1     # random forest size at every stage of boosting
  )
  
  config$xgb.n.rounds = 4000
  
  if (0) {
    warning('Runing quick single setup')
    config$xgb.tuned.pars$eta = 0.3
    config$xgb.n.rounds = 200
  }
  
  if (0) {
    warning('Running RF with XGB experiment')
    config$xgb.n.rounds = 1
    config$xgb.tuned.pars$eta               = 1
    config$xgb.tuned.pars$gamma             = 0
    config$xgb.tuned.pars$max_depth         = 50
    config$xgb.tuned.pars$min_child_weight  = 0
    config$xgb.tuned.pars$num_parallel_tree = 200
    config$xgb.tuned.pars$subsample         = 0.9
    config$xgb.tuned.pars$colsample_bytree  = 0.7
  }
  
  if (do.cv) {
    cat(date(), 'Doing CV of single XGB model\n')
    
    xgb.cv.res = xgb.cv(params = c(config$xgb.fixed.pars, config$xgb.tuned.pars), data = config$xgb.train.data, nrounds = config$xgb.n.rounds, prediction = T, folds = config$xgb.folds, feval = config$feval)
    
    plot(1:config$xgb.n.rounds, xgb.cv.res$dt$train.rmsle.mean, type = 'l', lty = 2, ylab = 'RMSLE', xlab = 'Boosting round', main = config$subtag, ylim = c(0.15, 0.3))
    lines(1:config$xgb.n.rounds, xgb.cv.res$dt$test.rmsle.mean + 2 * xgb.cv.res$dt$test.rmsle.std, lty = 3)
    lines(1:config$xgb.n.rounds, xgb.cv.res$dt$test.rmsle.mean)
    lines(1:config$xgb.n.rounds, xgb.cv.res$dt$test.rmsle.mean - 2 * xgb.cv.res$dt$test.rmsle.std, lty = 3)
    abline(h = 0.27, col = 2)
    abline(h = 0.22, col = 'orange')
    abline(h = 0.20, col = 3)
  }
  
  cat(date(), 'Training final single XGB model\n')
  x.mod.t = xgb.train(params = c(config$xgb.fixed.pars, config$xgb.tuned.pars), data = config$xgb.train.data, nrounds = config$xgb.n.rounds, feval = config$feval)
  test.preds = predict(x.mod.t, config$xgb.test.data) # NOTE: on the transformed response scale
  
  if (config$xgb.fixed.pars$booster != 'gblinear') {
    # Examine variable importance (takes some time to extract!)
    cat(date(), 'Examining importance of features in the single XGB model\n')
    impo = xgb.importance(colnames(model.matrix(~ . - cost - 1, data = config$train.data)), model = x.mod.t)
    #xgb.plot.importance(impo)
    print(impo[1:50, ])
  }
  
  if (do.sanity.check) {
    # NOTE: everything is on the transformed response scale
    valid.preds = predict(x.mod.t, config$xgb.valid.data)
    v1 = config$recover.response(config$valid.cost)
    v2 = config$recover.response(valid.preds)
    cat('\nSanity check RMSLE:', sqrt(mean((log(v1 + 1) - log(v2 + 1)) ^ 2)), '\n\n')
  }
} 

if (train.mode == 'bag.single') {
  # If we subsample heavily in the xgboost model, this means we can stack/bag
  # many such models (this was actually already suggested in the 0-24 script)
  
  config$xgb.tuned.pars = list(
    eta               = 0.05, # shrinkage along boosting rounds (lower will slow training convergence)
    gamma             = 0   , # single tree constraint (higher will tend to create less complex trees)
    max_depth         = 6   , # single tree constraint (effectively the maximum variable interactions per tree)
    min_child_weight  = 0   , # single tree constraint (higher will tend to create less complex trees)
    subsample         = 0.8 , # bagging-like randomization per round
    colsample_bytree  = 0.6 , # random-forest like randomization per round
    num_parallel_tree = 1     # random forest size at every stage of boosting
  )
  
  config$xgb.n.rounds = 4000
  
  config$bag.size = 8 # FIXME on condor, I'll ignore this and fit one model per job
  
  if (do.cv) {
    cat(date(), 'Doing CV of the bag of XGB models')
    
    cv.preds = rep(0, nrow(config$train.data))
    
    xgb.train.data = xgb.test.data = list()
    for (fi in 1:length(config$cv.folds)) {
      xgb.train.data[[fi]] = xgb.DMatrix(sparse.model.matrix(~ . - cost - 1, data = config$train.data[config$cv.folds[[fi]]$train, ]), missing = -999, label = config$train.data$cost[config$cv.folds[[fi]]$train])
      xgb.test.data [[fi]] = xgb.DMatrix(sparse.model.matrix(~ .        - 1, data = config$train.data[config$cv.folds[[fi]]$test , ]), missing = -999)
    }
    
    for (bi in 1:config$bag.size) {
      for (fi in 1:length(config$cv.folds)) {
        model = xgb.train(params = c(config$xgb.fixed.pars, config$xgb.tuned.pars), data = xgb.train.data[[fi]], nrounds = config$xgb.n.rounds, feval = config$feval, nthread = ifelse(config$compute.backend != 'serial', 1, 8))
        preds = predict(model, xgb.test.data[[fi]]) # NOTE: on the transformed response scale
        
        cv.preds[config$cv.folds[[fi]]$test] = ((bi - 1) * cv.preds[config$cv.folds[[fi]]$test] + preds) / bi
      }
      
      v1 = config$recover.response(cv.preds)
      v2 = config$recover.response(config$train.data$cost)
      cv.err = sqrt(mean((log(v1 + 1) - log(v2 + 1)) ^ 2))
      cat(date(), 'Bag iteration', bi, 'of', config$bag.size, '=> CV RMSLE:', cv.err, '\n')
    }
  }
  
  cat(date(), 'Training final bag of XGB models')
  
  bag.preds = compute.backend.run(
    config, config$process.bagged.xgb, combine = c, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.batch.name = 'IsoStack', 
    cluster.requirements = NULL
  )
  
  # NOTE: preds here are still on the transformed response scale
  test.preds = rowMeans(do.call(cbind, bag.preds[seq(2, length(bag.preds), 2)]))
  
  if (do.sanity.check) {
    valid.preds = rowMeans(do.call(cbind, bag.preds[seq(1, length(bag.preds), 2)]))
    v1 = config$recover.response(config$valid.cost)
    v2 = config$recover.response(valid.preds)
    cat('\nSanity check RMSLE:', sqrt(mean((log(v1 + 1) - log(v2 + 1)) ^ 2)), '\n\n')
  }
}

if (train.mode == 'tune.single') {
  if (0) {
    # Stage 1 of tuning - try various configerations (but not too many so as 
    # not to overfit)
    
    cat(date(), 'Tuning stage I\n')
    
    config$xgb.n.rounds = 3000 # limit this a bit
    
    grid.eta               = c(0.05, 0.01) # shrinkage along boosting rounds (lower will slow training convergence)
    grid.gamma             = c(0)          # single tree constraint (higher will tend to create less complex trees)
    grid.max_depth         = c(5)          # single tree constraint (effectively the maximum variable interactions per tree)
    grid.min_child_weight  = c(0)          # single tree constraint (higher will tend to create less complex trees)
    grid.subsample         = c(1)          # bagging-like randomization per round
    grid.colsample_bytree  = c(1)          # random-forest like randomization per round
    grid.num_parallel_tree = c(1)          # random forest size at every stage of boosting
  } else if (1) {
    # Stage 2 of tuning - tune the learning rate
    
    cat(date(), 'Tuning stage II\n')
    
    config$xgb.n.rounds = 10000
    
    grid.eta               = c(0.05, 0.01)
    grid.gamma             = 0
    grid.max_depth         = 5
    grid.min_child_weight  = 0
    grid.subsample         = 1
    grid.colsample_bytree  = 1
    grid.num_parallel_tree = 1
  } else {
    # Debug mode
    grid.eta               = c(0.5, 0.3)
    grid.gamma             = 0.02
    grid.max_depth         = 5
    grid.min_child_weight  = 1
    grid.subsample         = 0.7
    grid.colsample_bytree  = 0.7
    grid.num_parallel_tree = 1
  }
  
  config$xgb.tuned.pars.candidates = expand.grid(eta = grid.eta, gamma = grid.gamma, max_depth = grid.max_depth, min_child_weight = grid.min_child_weight, subsample = grid.subsample, colsample_bytree = grid.colsample_bytree, num_parallel_tree = grid.num_parallel_tree)
  
  cv.errs = compute.backend.run(
    config, config$process.tune.xgb, combine = c, 
    package.dependencies = config$package.dependencies,
    source.dependencies  = config$source.dependencies,
    cluster.dependencies = config$cluster.dependencies,
    cluster.batch.name = 'IsoStack', 
    cluster.requirements = NULL
  )
  
  cat('\nAll tuning setups:\n\n')
  idxs = order(cv.errs)
  print(cbind(cv.err = cv.errs, config$xgb.tuned.pars.candidates)[idxs, ])
  
  if (do.sanity.check) {
    # NOTE: all preds here are on the transformed response scale
    xgb.tuned.pars = config$xgb.tuned.pars.candidates[idxs[1], ]
    x.mod.t = xgb.train(params = c(config$xgb.fixed.pars, xgb.tuned.pars), data = config$xgb.train.data, nrounds = config$xgb.n.rounds, feval = config$feval)
    test.preds  = predict(x.mod.t, config$xgb.test.data )
    valid.preds = predict(x.mod.t, config$xgb.valid.data)
    v1 = config$recover.response(test.preds)
    v2 = config$recover.response(valid.preds)
    cat('\nSanity check RMSLE:', sqrt(mean((log(v1 + 1) - log(v2 + 1)) ^ 2)), '\n\n')
  }
}

if (train.mode == 'stack') {
  set.seed(config$rng.seed)
  config$L0.folds = get.atomic.random.folds(config$train.atomic.ids, config$nr.L0.folds)
  config$L1.folds = get.atomic.random.folds(config$train.atomic.ids, config$nr.L1.folds)
  config$L2.folds = get.atomic.random.folds(config$train.atomic.ids, config$nr.L2.folds)
  
  if (do.generate.l1.train.data) {
    cat(date(), 'Generating level 1 train data\n')
    
    set.seed(config$rng.seed)
    
    L1.res = compute.backend.run(
      config, config$process.L0, combine = rbind, 
      package.dependencies = config$package.dependencies,
      source.dependencies  = config$source.dependencies,
      cluster.dependencies = config$cluster.dependencies,
      cluster.batch.name = 'IsoStack', 
      cluster.requirements = NULL
    )
    
    L1.train.data = list(fold = L1.res[, 1], y = L1.res[, 2], X = L1.res[, -(1:2)])
    rm(L1.res)
    save(L1.train.data, file = '../Results/cater-L0.RData')
    
    # While we're at it, let's compute CV error for L0 models
    cv.rmsle = matrix(NA, config$nr.L0.folds, length(config$L0.model.names))
    for (i in 1:config$nr.L0.folds) {
      idx = (L1.train.data$fold == i)
      v1 = config$recover.response(L1.train.data$X[idx, ])
      v2 = config$recover.response(L1.train.data$y[idx])
      cv.rmsle[i, ] = sqrt(colMeans((log(v1 + 1) - log(v2 + 1)) ^ 2))
    }
    cv.rmsle = data.frame(cv.rmsle)
    names(cv.rmsle) = config$L0.model.names
    cat('\nCross validation RMSLE of L0 models:\n\n')
    print(cv.rmsle)
    cat('\nCross validation means RMSLE of L0 models:\n\n')
    print(colMeans(cv.rmsle))
    cat('\n')
    boxplot(cv.rmsle)
    abline(h = 0.27, col = 2)
    abline(h = 0.22, col = 'orange')
    abline(h = 0.20, col = 3)
  }
  
  if (do.generate.l1.test.data) {
    cat(date(), 'Generating level 1 test data\n')
    
    set.seed(config$rng.seed)
    
    L0.fitted.models = config$train.L0(config, config$train.data)
    L0.preds = config$predict.L0(config, L0.fitted.models, config$test.data)
    L1.test.X = data.matrix(L0.preds)
    
    save(L1.test.X, file = '../Results/cater-L0-test.RData')
  }
  
  if (do.generate.l2.train.data) {
    cat(date(), 'Generating level 2 train data\n')
    
    load('../Results/cater-L0.RData') # => L1.train.data
    config$L1.train.data = L1.train.data
    rm(L1.train.data)
    
    set.seed(config$rng.seed)
    
    L2.res = compute.backend.run(
      config, config$process.L1, combine = rbind, 
      package.dependencies = config$package.dependencies,
      source.dependencies  = config$source.dependencies,
      cluster.dependencies = config$cluster.dependencies,
      cluster.batch.name = 'IsoStack', 
      cluster.requirements = NULL
    )
    
    L2.train.data = list(fold = L2.res[, 1], y = L2.res[, 2], X = L2.res[, -(1:2)])
    rm(L2.res)
    save(L2.train.data, file = '../Results/cater-L1.RData')
    
    # While we're at it, let's compute CV error for L1 models
    # NOTE: this is not percise, since we ignore the CV done in L0
    cv.rmsle = matrix(NA, config$nr.L1.folds, length(config$L1.model.names))
    for (i in 1:config$nr.L1.folds) {
      idx = (L2.train.data$fold == i)
      v1 = config$recover.response(L2.train.data$X[idx, ])
      v2 = config$recover.response(L2.train.data$y[idx])
      cv.rmsle[i, ] = sqrt(colMeans((log(v1 + 1) - log(v2 + 1)) ^ 2))
    }
    cv.rmsle = data.frame(cv.rmsle)
    names(cv.rmsle) = config$L1.model.names
    cat('\nCross validation RMSLE of L1 models:\n\n')
    print(cv.rmsle)
    cat('\nCross validation means RMSLE of L1 models:\n\n')
    print(colMeans(cv.rmsle))
    cat('\n')
    boxplot(cv.rmsle)
    abline(h = 0.27, col = 2)
    abline(h = 0.22, col = 'orange')
    abline(h = 0.20, col = 3)
  }
  
  if (do.generate.l2.test.data) {
    cat(date(), 'Generating level 2 test data\n')
    
    load('../Results/cater-L0.RData') # => L1.train.data
    load('../Results/cater-L0-test.RData') # => L1.test.X
    
    set.seed(config$rng.seed)
    
    if (config$use.irp) {
      old.wd = getwd()
      tmp.wd = paste0('tmp/core', 1)
      dir.create(tmp.wd, showWarnings = F)
      file.copy('irp_wrapper.m', tmp.wd, overwrite = T)
      setwd(tmp.wd)
      matlab.port = 10000 + 1
      Matlab$startServer(port = matlab.port)
      config$matlab = Matlab(port = matlab.port)
      isOpen = open(config$matlab)
    }
    
    L1.fitted.models = config$train.L1(config, data.matrix(L1.train.data$X), L1.train.data$y)
    L1.preds = config$predict.L1(config, L1.fitted.models, L1.test.X)
    L2.test.X = data.matrix(L1.preds)
    
    if (config$use.irp) {
      close(config$matlab)
      setwd(old.wd)
    }
    
    save(L2.test.X, file = '../Results/cater-L1-test.RData')
    
    test.preds = L2.test.X[, match('IAM', config$L1.model.names)]
  }
  
  if (do.generate.l3.train.data) {
    cat(date(), 'Generating level 3 train data\n')
    
    load('../Results/cater-L1.RData') # => L2.train.data
    config$L2.train.data = L2.train.data
    rm(L2.train.data)
    
    set.seed(config$rng.seed)
    
    L3.res = compute.backend.run(
      config, config$process.L2, combine = rbind, 
      package.dependencies = config$package.dependencies,
      source.dependencies  = config$source.dependencies,
      cluster.dependencies = config$cluster.dependencies,
      cluster.batch.name = 'IsoStack', 
      cluster.requirements = NULL
    )
    
    L3.train.data = list(fold = L3.res[, 1], y = L3.res[, 2], X = L3.res[, -(1:2)])
    rm(L3.res)
    save(L3.train.data, file = '../Results/cater-L2.RData')
    
    # While we're at it, let's compute CV error for L2 models
    # NOTE: this is not percise, since we ignore the CV done in L0, L1
    cv.rmsle = matrix(NA, config$nr.L2.folds, length(config$L2.model.names))
    for (i in 1:config$nr.L2.folds) {
      idx = (L3.train.data$fold == i)
      v1 = config$recover.response(L3.train.data$X[idx, ])
      v2 = config$recover.response(L3.train.data$y[idx])
      cv.rmsle[i, ] = sqrt(colMeans((log(v1 + 1) - log(v2 + 1)) ^ 2))
    }
    cv.rmsle = data.frame(cv.rmsle)
    names(cv.rmsle) = config$L2.model.names
    cat('\nCross validation RMSLE of L2 models:\n\n')
    print(cv.rmsle)
    cat('\nCross validation means RMSLE of L2 models:\n\n')
    print(colMeans(cv.rmsle))
    cat('\n')
    boxplot(cv.rmsle)
    abline(h = 0.27, col = 2)
    abline(h = 0.22, col = 'orange')
    abline(h = 0.20, col = 3)
  }
  
  if (do.generate.l3.test.data) {
    cat(date(), 'Generating level 3 test data\n')
    
    load('../Results/cater-L1.RData') # => L2.train.data
    load('../Results/cater-L1-test.RData') # => L2.test.X
    
    set.seed(config$rng.seed)
    
    if (config$use.irp) {
      old.wd = getwd()
      tmp.wd = paste0('tmp/core', 1)
      dir.create(tmp.wd, showWarnings = F)
      file.copy('irp_wrapper.m', tmp.wd, overwrite = T)
      setwd(tmp.wd)
      matlab.port = 10000 + 1
      Matlab$startServer(port = matlab.port)
      config$matlab = Matlab(port = matlab.port)
      isOpen = open(config$matlab)
    }
    
    L2.fitted.models = config$train.L2(config, data.matrix(L2.train.data$X), L2.train.data$y)
    L2.preds = config$predict.L2(config, L2.fitted.models, L2.test.X)
    L3.test.X = data.matrix(L2.preds)
    
    if (config$use.irp) {
      close(config$matlab)
      setwd(old.wd)
    }
    
    save(L3.test.X, file = '../Results/cater-L2-test.RData')
    
    test.preds = L3.test.X[, match('NNLS', config$L2.model.names)]
  }
}

if (do.generate.submission) {
  test.yp = config$recover.response(test.preds)
  
  cat(date(), 'Generating submission\n')
  submitDb = data.frame(id = config$test.ids, cost = test.yp)
  submitDb = aggregate(data.frame(cost = submitDb$cost), by = list(id = submitDb$id), mean)
  write.csv(submitDb, paste0('../Results/cater-submit-', config$tag, '.csv'), row.names = F, quote = F)
  save(config, file = paste0('../Results/cater-submit-', config$tag, '.RData'))
  
  if (1) {
    # Compare to the best submission so far (as just another sanity check)
    new.submitDb = submitDb
    new.config = config
    load('../Results/cater-submit-T5.RData') # => config
    old.config = config
    rm (config)
    old.submitDb = read.csv('../Results/cater-submit-T5.csv', header = T)
    stopifnot(all(old.submitDb$id == new.submitDb$id))
    plot(old.submitDb$cost, new.submitDb$cost, xlab = 'best submission', ylab = 'new submission', log = 'xy')
  }
}

cat(date(), 'Done.\n')
