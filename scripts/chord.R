library(ggplot2)
library(circlize)
library(ggsci)
library(ggthemes)
library(rcartocolor)

na <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/north_america_migrations.csv"
sa <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/south_america_migrations.csv"
af <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/africa_migrations.csv"
as <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/asia_migrations.csv"
oc <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/oceania_migrations.csv"
eu <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/europe_migrations.csv"
al <- "C:/Users/Sean/Documents/MATH_498/code/generated_data/all_migrations.csv"

mat <- read.table(na, sep = '\t', header = TRUE, stringsAsFactors = FALSE)

country_size <- function(m, c) {
  froms <- sum(m$Migrations[m$From == c])
  tos <- sum(m$Migrations[m$To == c])
  return(froms + tos)
}

top_countries <- function(s, l) {
  s <- sort(s, decreasing = TRUE)
  return(names(s)[1:l])
}

limit = 12
#na 10000, sa 500, af 1000, as 1000, oc 500, eu 10000
min_flow = 10000

countries <- unique(mat$From)
sizes = vapply(countries, function(x) {country_size(mat, x)}, FUN.VALUE = c(0))
names(sizes) <- countries
keepers = top_countries(sizes, limit)
mat <- mat[mat$From %in% keepers,]
mat <- mat[mat$To %in% keepers,]
mat <- mat[mat$Migrations > min_flow,]

mat$Migrations <- mat$Migrations / 1000
#cols = rainbow(length(unique(mat$From)))
#cols = gdocs_pal()(length(unique(mat$From)))
cols = carto_pal((length(unique(mat$From))), "Prism")
names(cols) <- unique(mat$From)
circos.par(gap.after = 2.5, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)
chordDiagram(mat, directional = -1, grid.col = cols, transparency = 0.3,
             diffHeight = uh(5, "mm"), annotationTrack = "grid",
             annotationTrackHeight = uh(7, "mm"), 
             preAllocateTracks = list(track.height = max(strwidth(unlist(mat$From)))))
for(si in get.all.sector.index()) {
  circos.axis(h = "top", labels.cex = 1, labels.font = 2, sector.index = si, track.index = 2,
              minor.ticks = 1)
}
circos.track(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  xplot = get.cell.meta.data("xplot")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  
  if(abs(xplot[2] - xplot[1]) < 15) {
    fac <- "clockwise"
    ad <- c(-1, 0.5)
  } else {
    fac <- "bending"
    ad <- c(0.5, -2)
  }
  circos.text(mean(xlim), ylim[1], sector.name, facing = fac, font = 2,
              niceFacing = TRUE, adj = ad, col= cols[CELL_META$sector.numeric.index])
}, bg.border = NA)
circos.clear()
