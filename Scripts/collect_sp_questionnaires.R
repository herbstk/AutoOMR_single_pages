#!/usr/bin/env Rscript
library(tidyverse)

scans_dir = commandArgs(trailingOnly=TRUE)[[1]]

if(!dir.exists(scans_dir))
  stop(str_c("Directory not found: ", scans_dir))

COLS <- c("answer_id", "state", "fill", "bgr")
COLS_TYPES <- "cddd"

collect_scan_results <- function(path){
  processed_files <- list.files(path, pattern = "\\.tsv", recursive = TRUE, full.names = TRUE)
  processed_files <- processed_files[!str_detect(processed_files, "results.tsv")]
  
  scan_results <- lapply(processed_files, function(.){
    ret <- read_tsv(., col_names = COLS, col_types = COLS_TYPES)
    meta <- str_match(., "/([0-9]+)-([0-9]+)-(.+)\\.tsv$")
    ret$filename <- .
    ret$scan_date <- meta[1,2]
    ret$scan_time <- meta[1,3]
    ret$hash6 <- meta[1,4]
    answer_ids <- str_split_fixed(ret$answer_id, "_", 5)
    ret$q_type <- answer_ids[,1]
    ret$q_id <- answer_ids[,2]
    ret$q_nr <- as.numeric(answer_ids[,3])
    ret$q_option_nr <- as.numeric(answer_ids[,4])
    ret$q_option <- answer_ids[,5]
    ret
  }) %>%
    bind_rows() %>%
    mutate(q_id = as.numeric(str_extract(q_id, "\\d+")))%>%
    arrange(scan_date, scan_time, q_id, q_nr)
  return(scan_results)
}

aggregate_answers <- function(scan_results){
  scan_results_out <- scan_results %>%
    filter( q_type == "CB" ) %>% # discard text-box answers
    filter( q_option != "other" ) %>% # remove answers which are uninformative
    group_by(scan_date, scan_time, hash6, q_id) %>%
    mutate(answer = case_when(state == 1 ~ q_option,
                              state == -1 ~ "?",
                              TRUE ~ "") ) %>%
    summarise(answers = sum(answer != ""),
              answer = str_c(answer, collapse = "")) %>%
    ungroup() %>%
    mutate(answer = if_else(answer == "", "unanswered", answer))
  return(scan_results_out)
}

print(str_c("Processing: ", scans_dir))
result <- collect_scan_results(scans_dir)
result %>% write_tsv(file.path(scans_dir, "gathered_results.tsv"))

result_aggr <- aggregate_answers(result)
## double check multiple answers
result_aggr %>% filter(answers > 1) %>% print
result_aggr %>% write_tsv(file.path(scans_dir, "aggregated_results.tsv"))

