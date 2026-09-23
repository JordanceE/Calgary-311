# Final GitHub и Moodle checklist

## Пред final push

- Провери дека `README.md`, `requirements.txt`, `Calgary_311_Project.ipynb` и целата папка `code/` се вклучени.
- Провери дека `data/raw/` и `data/processed/` се вклучени. Сите тековни датотеки се под GitHub лимитот од 100 MB.
- Провери дека `paper/Calgary_311_seminarska_rabota.pdf` и `presentation/Calgary_311_presentation.pptx` се вклучени.
- `.venv/`, `.idea/`, `__pycache__/` и notebook checkpoints не треба да се commit-ираат.
- Ако repo-то е private, додај го корисникот `BiljanaTR` како collaborator.

## Final commit и push

Од коренот на проектот:

```powershell
git status
git add .
git status
git commit -m "Final project submission"
git push origin main
git rev-parse HEAD
```

Последната команда го прикажува commit hash-от што треба да се внесе во Moodle. Пред commit, внимателно провери го вториот `git status` за да нема лични, привремени или непотребни датотеки.

## Moodle Online text

```text
Наслов: Calgary 311 Service Requests and Historical Weather Analysis
Членови: [имиња на сите членови]
GitHub repo: [линк до repo]
Final commit: [целосен commit hash]
```

Во Moodle одделно прикачи ги конечниот PDF извештај и конечната презентација во PDF или PowerPoint формат.
