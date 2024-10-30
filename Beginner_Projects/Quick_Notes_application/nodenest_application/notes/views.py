from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from .models import Note
from .forms import NoteForm


def home(request):
    return render(request, "home.html")


class NoteListView(View):
    def get(self, request):
        notes = Note.objects.all()
        return render(request, "note_list.html", {"notes": notes})


class NoteCreateView(View):
    def get(self, request):
        form = NoteForm()
        return render(request, "note_form.html", {"form": form})

    def post(self, request):
        form = NoteForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("note_list")
        return render(request, "note_form.html", {"form": form})


class NoteUpdateView(View):
    def get(self, request, pk):
        note = get_object_or_404(Note, pk=pk)
        form = NoteForm(instance=note)
        return render(request, "note_form.html", {"form": form})

    def post(self, request, pk):
        note = get_object_or_404(Note, pk=pk)
        form = NoteForm(request.POST, instance=note)
        if form.is_valid():
            form.save()
            return redirect("note_list")
        return render(request, "note_form.html", {"form": form})


class NoteDeleteView(View):
    def get(self, request, pk):
        note = get_object_or_404(Note, pk=pk)
        return render(request, "note_confirm_delete.html", {"note": note})

    def post(self, request, pk):
        note = get_object_or_404(Note, pk=pk)
        note.delete()
        return redirect("note_list")
