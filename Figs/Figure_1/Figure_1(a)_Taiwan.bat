gmt begin Figure_1(a)_Taiwan png 
	gmt set FONT_ANNOT_PRIMARY 14p,Helvetica,black
    gmt coast -JM10c -R118/123/21/27 -ETW+gred -Ba1f1 -G244/243/239 -S167/194/223  
	gmt plot CN-border-L1.gmt -W0.5p
	echo 120.26 22.73 | gmt plot -Sr0.5c -Gblack
	echo 120.26 22.73 Tsukuan| gmt text -F+f12p,1,black+jTL -D0c/0.6c
	echo 121.87 24.91 | gmt plot -Sr0.5c -Gblack
	echo 121.87 24.91 Kengfang| gmt text -F+f12p,1,black+jTL -D-2c/-0.38c

	echo 120.9 26.3 East China Sea| gmt text -F+f14p,2,red+jTL -D0c/0.6c 
	echo 118.5 21.2 South China Sea| gmt text -F+f14p,2,red+jTL -D0c/0.6c

	gmt plot locations.dat -St0.3c -W2p,blue
	gmt image ./LH_mask.png -Dg121/25.4+w-3.6
	gmt image ./LH_mask.png -Dg118.5/22.4+w-3.0
	
	gmt inset begin -DjBR+w1.3i+o0.1i/0.1i -F+gwhite+p1p+c0.1c
	gmt set MAP_GRID_PEN_PRIMARY 25p,black,2_2
		gmt coast -JG120/24N/? -Rg -Bg -Wfaint -G244/243/239 -S167/194/223 -ETW+gred -A5000
		gmt plot CN-border-L1.gmt -W0.2p
		echo 118 21 123 26 | gmt plot -Sr+s -W1p,red
    gmt inset end

	@REM gmt inset begin -DjLB+w2.6c/2.15c -F+p0.5p
	@REM gmt coast -JM? -R70/138/13/56 -G244/243/239 -S167/194/223 -Df
	@REM gmt plot CN-border-L1.gmt -W0.2p
	@REM echo 118 21 123 26 | gmt plot -Sr+s -W1p,red
    @REM gmt inset end
gmt end show
