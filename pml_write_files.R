pml_write_files = function(x){
    if( !(file.exists("test_results")) ){
        dir.create("test_results")
    }
    n = length(x)
    for(i in 1:n){
        filename = paste0("test_results/problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}