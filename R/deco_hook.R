library(knitr)


knit_hooks$set(decorate = function(before, options) {
  if (before) {
     if (options$engine == "python") {
       label <- "<b>Python</b>"
       bc <- "#417FB1"
       sz <- "90%"
       tc <- "#FFD94C"
       icon <- "<i class=\"fab fa-python\"></i>  "
                      
  
     } else if (options$engine == "R") {
       label <- "<b>R</b>"
       bc <- "#4C78DB"
       sz <- "90%"
       tc <- "#ffffff"
       icon <- "<i class=\"fab fa-r-project\"></i>  "
       
     } else if (options$engine == "bash") {
       label <- "<b>Shell</b>"
       bc <- "#000000"
       sz <- "90%"
       tc <- "#ffffff"
       icon <- "<i class=\"fas fa-terminal\"></i>  "
       
     }
    paste0("<div class=decocode>",
           "<div style=\"background-color:", bc, "\">",
           "<span style=\"font-size:", sz, ";color:", tc, "\">",
           icon, label, "</span>", "\n")
  } else {
    "</div><br></div>"
  }
  
}) 



# knit_hooks$set(deco = function(before, options) {
#   if (before) {
#     label <- ifelse(is.null(options$deco$label), "<b>Python</b>",
#                     paste0("<b>", options$deco$label, "</b>"))
#     # bc <- ifelse(is.null(options$deco$bc), "#366994", options$deco$bc)
#     bc <- ifelse(is.null(options$deco$bc), "#417FB1", options$deco$bc)
#     sz <- ifelse(is.null(options$deco$sz), "90%", options$deco$sz)
#     tc <- ifelse(is.null(options$deco$tc), "#FFD94C", options$deco$tc)
#     icon <- ifelse(is.null(options$deco$icon),
#                    "<i class=\"fab fa-python\"></i>  ",
#                    paste0("<i class=\"", options$deco$icon$style,
#                           " fa-", options$deco$icon$name, "\"></i>  "))
#     paste0("<div class=decocode>",
#            "<div style=\"background-color:", bc, "\">",
#            "<span style=\"font-size:", sz, ";color:", tc, "\">",
#            icon, label, "</span>")
#   } else {
#     "</div></div>"
#   }
# })
