


from pathlib import Path


if __name__ == "__main__":
	if Path.cwd() / "data" / "features_extracted.pkl".exists():
		print("Features already extracted") 