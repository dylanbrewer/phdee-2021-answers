clear all
set more off

* Set directories

	local datapath = "C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework9"
	local outputpath = "C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-answers\homework9\output"
	
* Load data

	cd "`datapath'"
	import delimited fishbycatchupdated.csv, clear
	
* Plot scheme

	set scheme plotplainblind
	
* Clean data

	reshape long shrimp salmon bycatch, i(firm) j(month)
	
	gen didtreatment = treated
	replace didtreatment = 0 if month < 13
	replace didtreatment = 1 if month > 24 // staggered adoption
	
* Question 1

	cd "`outputpath'"
	preserve // Saves the current workspace, allowing me to manipulate the data to plot averages
		collapse (mean) bycatch, by(month treated) // Collapses to the mean by month and by treatment group
		twoway (line bycatch month if treated == 1, xline(12.5 24.5) legend(order(1 "First treated" 2 "Second treated") position(1) ring(0))) (line bycatch month if treated == 0)
		graph export hw9q1.eps, replace
	restore
	
* Question 2

	xtset firm month
	
	xtreg bycatch didtreatment i.month shrimp salmon, fe vce(cluster firm)
	
	eststo q2xtreg
	
* Question 3
	
	reghdfe bycatch didtreatment shrimp salmon, a(i.month i.firm) vce(cluster month firm)
	
	eststo q3reghdfe
	
* Question 5
	
	twowayfeweights bycatch firm month didtreatment, type(feTR) controls(shrimp salmon)
	
* Question 6
	
	capture program drop didm // It turns out that DID_M is not programmed like a typical stata command, so we needed a wrapper around it to return results in the e-class that play well with other Stata commands
	program define didm, eclass
		tempname b V N nobs
		did_multiplegt bycatch firm month didtreatment, controls(shrimp salmon) cluster(firm) breps(50)

		matrix `b' = nullmat(`b'), `e(effect_0)'
		matrix rownames `b' = DIDM
		matrix colnames `b' = didtreatment
		matrix `V' = nullmat(`V'), `e(se_effect_0)'^2
		matrix `V' = diag(`V')
		matrix rownames `V' = didtreatment
		matrix colnames `V' = didtreatment
		local nobs = e(N_effect_0)
		
		qui: reg bycatch didtreatment
		
		ereturn post `b' `V'
		ereturn scalar N = `nobs'
	end
	
	didm // There are probably more elegant (and certainly more programmatically flexible) ways to do this, but this works.
	
	eststo q6didm
	
* Question 7

	preserve
		drop if month > 24
		twowayfeweights bycatch firm month didtreatment, type(feTR) controls(shrimp salmon)
	restore
	
* Plots

	la var didtreatment "Treatment"
	la var shrimp "Shrimp"
	la var salmon "Salmon"

	esttab q2xtreg q3reghdfe q6didm using didtable.tex, tex keep(didtreatment shrimp salmon) label stats(N) mtitle(xtreg reghdfe DID_M) replace