#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:55:14 2019

@author: timur
"""

import os # probably deprecated
import subprocess
import re
import numpy as np
import pandas as pd
import re
import urllib
import shlex
import glob
#study summury file
# eg https://www.ebi.ac.uk/ena/data/view/PRJEB15062
log_file = "/home/timur/PIPELINE_LOG.txt"
log_file_h = open(log_file, "a")
log_file_h.writelines("Starting Script.... \n")

study_sum_file = "/home/timur/ext/DBs/data_sets/PRJEB15062_iPS.txt"

working_dir = os.path.join("/home","timur","ext","working_dir","PRJEB15062")
download_dir = os.path.join(working_dir, "raw_fastas")

fasta_dir = os.path.join(working_dir, "fastas")
bam_dir = os.path.join(working_dir, "BAM")
brie_dir = os.path.join(working_dir,"BRIE_output")

trim_galore_path = "/home/timur/TrimGalore-0.6.0/trim_galore"
trim_galore_pars = "-j 4"

STAR_pars = " --runThreadN 10 --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outFilterType BySJout --outFilterMultimapNmax 20 --alignSJoverhangMin 8 --alignSJDBoverhangMin 1 --outFilterMismatchNmax 999 --alignIntronMin 20 --alignIntronMax 1000000 --alignMatesGapMax 1000000 --twopassMode Basic"
STAR_genome_dir = "/home/timur/ext/DBs/STAR_gen_index"

BRIE_pars = " -f /home/timur/ext/DBs/BRIE/factors/human_factors.SE.filtered.csv.gz -a /home/timur/ext/DBs/BRIE/factors/SE.filtered.gtf.gz"
#https://brie-rna.sourceforge.io/manual.html

def run_shell_command(cmd):
    print("try to execute:\n" + cmd)
    args = shlex.split(cmd)
    compl_proc = subprocess.run(args)
    out = compl_proc.stdout
    exit_status = compl_proc.returncode
    print("Exit status: " + str(exit_status))
    print(out)
    return exit_status
#os.system("mkdir " + working_dir)
try:
    os.makedirs(working_dir, exist_ok = True)
    os.makedirs(download_dir, exist_ok = True)
    os.makedirs(fasta_dir, exist_ok = True)
    os.makedirs(bam_dir, exist_ok = True)
    os.makedirs(brie_dir, exist_ok = True)
    
except Exception as err:
    print(err)


study_data = pd.read_table(study_sum_file, sep="\\t")
study_data["fastq_ftp"]
filtered_study_data = study_data[study_data.library_selection == "cDNA"]
list(filtered_study_data)

ext_pattern = re.compile("(\\..*)")
for index, row in   filtered_study_data.iterrows():
    cell_name = row.experiment_alias
    log_file_h.writelines("PROCEED CELL: %s \n" % cell_name)
    ftp_files = row.fastq_ftp.split(";")
    #download files
    
    d_files = []
    for i, ftp_file in enumerate(ftp_files):
#        f_name, extension = os.path.splitext(ftp_file) # extracts only last ext
        re_res = re.search(ext_pattern, os.path.basename(ftp_file))
        extension = re_res.group(0)
        down_file_name = cell_name + "_" + str(i+1) + extension
        down_file_name = os.path.join(download_dir, down_file_name)
        d_files.append(down_file_name)
        if( not os.path.exists(down_file_name)):
            log_file_h.writelines("Download cell: %s \n" % cell_name)
            urllib.request.urlretrieve("ftp://" + ftp_file, down_file_name)
#    tr_gal_path = os.path.join(fasta_dir, cell_name)
            
    #Trim Galore SECTIOn
    tr_gal_cmd = trim_galore_path
    if(len(ftp_files) == 2):
        tr_gal_cmd = tr_gal_cmd + " --paired"
    tr_gal_cmd += " " +  " ".join(d_files)
    tr_gal_cmd += " " + "-o " + fasta_dir 
#    tr_gal_cmd += " " + "--basename " + cell_name
    tr_gal_cmd +=  " " + trim_galore_pars
    
    #existing files in fasta dir
    trimmed_files = glob.glob(os.path.join(fasta_dir, cell_name + "_*.fq*"))
    if(len(trimmed_files) < len(d_files)):
        log_file_h.writelines("TRIMMING CELL: %s \n" % cell_name)
#        print("Trim Galore cmd: " + tr_gal_cmd)
        run_shell_command(tr_gal_cmd)
    trimmed_files = glob.glob(os.path.join(fasta_dir, cell_name + "_*.fq*"))
    
    # STAR SECTION
    bam_file = glob.glob(os.path.join(bam_dir, cell_name + "Aligned*.bam"))
    if(len(bam_file) < 1):
        log_file_h.writelines("CREATE BAM (STAR): %s \n" % cell_name)
        star_cmd = "STAR --genomeDir " + STAR_genome_dir
        star_cmd += " --readFilesIn " + " ".join(trimmed_files)
        star_cmd += " --outFileNamePrefix " + os.path.join(bam_dir, cell_name)
        star_cmd += " " + STAR_pars
#        print("STAR cmd: " + star_cmd)
        run_shell_command(star_cmd)
    bam_file = glob.glob(os.path.join(bam_dir, cell_name + "Aligned*.bam"))
    
    #SAMTOOLS INDEXING SECTION
    if(len(bam_file) ==1):
        sam_cmd = "samtools index " + bam_file[0]
        run_shell_command(sam_cmd)
    # BRIE SECTION
        brie_cmd = "brie  -s " + bam_file[0]
        brie_cmd += " -o " + os.path.join(brie_dir, cell_name)
        brie_cmd += " " + BRIE_pars
        if(not os.path.exists(os.path.join(brie_dir, cell_name))):
            log_file_h.writelines("RUN BRIE: %s \n" % cell_name)
            run_shell_command(brie_cmd)
        
log_file_h.close()
