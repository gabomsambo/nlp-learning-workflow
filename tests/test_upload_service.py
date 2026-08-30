"""Tests for upload service metadata enrichment and vector storage."""

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from nlp_pillars.schemas import PaperRef
from nlp_pillars.services.upload_service import (
    PipelineOutcome,
    UploadError,
    UploadService,
)


@pytest.fixture
def upload_service():
    """Create an upload service instance for testing."""
    return UploadService()


class TestArxivIdExtraction:
    """Test arXiv ID detection from various URL and filename formats."""

    def test_arxiv_id_from_abs_url(self):
        """Test arXiv ID detection from abs URL."""
        url = "https://arxiv.org/abs/2301.00001"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_pdf_url(self):
        """Test arXiv ID detection from PDF URL."""
        url = "https://arxiv.org/pdf/2301.00001.pdf"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_versioned_url(self):
        """Test arXiv ID detection from versioned URL."""
        url = "https://arxiv.org/abs/2301.00001v2"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_from_filename(self):
        """Test arXiv ID detection from filename."""
        filename = "2301.00001.pdf"
        match = re.search(r'(\d{4}\.\d{4,5})', filename)
        assert match is not None
        assert match.group(1) == "2301.00001"

    def test_arxiv_id_5_digit_minor(self):
        """Test arXiv ID with 5-digit minor version."""
        url = "https://arxiv.org/abs/2301.12345"
        match = re.search(r'(\d{4}\.\d{4,5})', url)
        assert match is not None
        assert match.group(1) == "2301.12345"


class TestArxivEnrichment:
    """Test arXiv metadata enrichment."""

    @patch('nlp_pillars.services.paper_metadata.arxiv.Client')
    @patch('nlp_pillars.services.paper_metadata.arxiv.Search')
    def test_arxiv_enrichment_success(
        self, mock_search_class, mock_client_class, upload_service
    ):
        """Test successful arXiv metadata enrichment."""
        # Mock arXiv response
        mock_result = Mock()
        mock_result.title = "Attention Is All You Need"
        mock_result.authors = [Mock(name="Vaswani"), Mock(name="Shazeer")]
        mock_result.published = Mock(year=2017)
        mock_result.summary = "The dominant sequence transduction models..."
        mock_result.journal_ref = None
        mock_result.primary_category = "cs.CL"

        mock_client = Mock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_class.return_value = mock_client

        paper = PaperRef(id="1706.03762", title="", authors=[], url_pdf="")
        enriched = upload_service._enrich_from_arxiv(paper, "1706.03762")

        assert enriched.title == "Attention Is All You Need"
        assert len(enriched.authors) == 2

    @patch('nlp_pillars.services.paper_metadata.arxiv.Client')
    @patch('nlp_pillars.services.paper_metadata.arxiv.Search')
    def test_arxiv_enrichment_uses_api_data(
        self, mock_search_class, mock_client_class, upload_service
    ):
        """Test that arXiv enrichment uses API data (authoritative source)."""
        mock_result = Mock()
        mock_result.title = "API Title"
        mock_result.authors = [Mock(name="Author")]
        mock_result.published = Mock(year=2020)
        mock_result.summary = "Abstract..."
        mock_result.journal_ref = None
        mock_result.primary_category = "cs.AI"

        mock_client = Mock()
        mock_client.results.return_value = iter([mock_result])
        mock_client_class.return_value = mock_client

        paper = PaperRef(
            id="2301.00001",
            title="PDF Extracted Title",  # Will be overwritten by API
            authors=[],
            url_pdf=""
        )
        enriched = upload_service._enrich_from_arxiv(paper, "2301.00001")

        # arXiv API data is authoritative - should be used
        assert enriched.title == "API Title"
        assert enriched.year == 2020

    def test_arxiv_enrichment_non_arxiv_paper(self, upload_service):
        """Test that non-arXiv papers are returned unchanged."""
        paper = PaperRef(id="some-other-id", title="Test", authors=[], url_pdf="")
        result = upload_service._enrich_from_arxiv(paper, "https://example.com/paper.pdf")
        assert result.title == "Test"


class TestS2Enrichment:
    """Test Semantic Scholar metadata enrichment."""

    def test_s2_enrichment_method_exists(self, upload_service):
        """Test that S2 enrichment method exists and is callable."""
        assert hasattr(upload_service, '_enrich_from_semantic_scholar')
        assert callable(upload_service._enrich_from_semantic_scholar)

    def test_s2_enrichment_with_short_title(self, upload_service):
        """Test that S2 enrichment skips short titles."""
        # Paper with very short title should not trigger S2 search
        paper = PaperRef(
            id="non-arxiv-id",
            title="Short",  # Too short for S2 search
            authors=[],
            url_pdf=""
        )
        # This should return without error (may not enrich if S2 fails)
        result = upload_service._enrich_from_semantic_scholar(paper)
        # Paper should still be returned unchanged (short title)
        assert result.id == "non-arxiv-id"

    @patch('nlp_pillars.services.paper_metadata.enrich_from_semantic_scholar')
    def test_s2_enrichment_preserves_existing_data(self, mock_enrich, upload_service):
        """Test that S2 enrichment preserves existing user data."""
        paper = PaperRef(
            id="non-arxiv-id",
            title="User Title",
            authors=["User Author"],  # Already has authors
            year=2024,  # Already has year
            url_pdf=""
        )
        mock_enrich.return_value = paper
        # Even if S2 returns data, user data should be preserved
        result = upload_service._enrich_from_semantic_scholar(paper)
        assert result.authors == ["User Author"]
        assert result.year == 2024


class TestTitleSimilarity:
    """Test title similarity matching."""

    def test_titles_identical(self, upload_service):
        """Test that identical titles are similar."""
        assert upload_service._titles_similar(
            "Attention Is All You Need",
            "Attention Is All You Need"
        )

    def test_titles_case_insensitive(self, upload_service):
        """Test that title comparison is case insensitive."""
        assert upload_service._titles_similar(
            "Attention Is All You Need",
            "attention is all you need"
        )

    def test_titles_with_punctuation(self, upload_service):
        """Test that punctuation is ignored."""
        assert upload_service._titles_similar(
            "Attention Is All You Need!",
            "Attention Is All You Need"
        )

    def test_titles_different(self, upload_service):
        """Test that different titles are not similar."""
        assert not upload_service._titles_similar(
            "Attention Is All You Need",
            "BERT: Pre-training of Deep Bidirectional Transformers"
        )


class TestVectorStorage:
    """Test vector storage integration."""

    @patch('nlp_pillars.services.upload_service.vectors.ensure_collections')
    @patch('nlp_pillars.services.upload_service.vectors.upsert_text')
    def test_vectors_module_import(self, mock_upsert, mock_ensure):
        """Test that vectors module can be imported."""
        from nlp_pillars import vectors
        assert hasattr(vectors, 'ensure_collections')
        assert hasattr(vectors, 'upsert_text')


class TestPaperIdGeneration:
    """Test paper ID generation from URLs and filenames."""

    def test_arxiv_url_generates_arxiv_id(self, upload_service):
        """Test that arXiv URLs generate proper arXiv IDs."""
        url = "https://arxiv.org/abs/2301.00001"
        paper_id = upload_service._generate_paper_id_from_url(url)
        assert paper_id == "arxiv:2301.00001"

    def test_non_arxiv_url_generates_hash_id(self, upload_service):
        """Test that non-arXiv URLs generate hash-based IDs."""
        url = "https://example.com/paper.pdf"
        paper_id = upload_service._generate_paper_id_from_url(url)
        assert paper_id.startswith("url:")
        assert len(paper_id) > 5


class TestUploadedPdfRetention:
    """Uploaded PDFs must outlive the request that created them.

    ``_create_paper_ref_from_file`` stores ``url_pdf = file://<abs path>``, and
    podcast generation dereferences that path to extract the paper body. The
    upload handler used to delete the file in a ``finally:`` block, so every
    file-uploaded paper permanently pointed at a path that no longer existed
    and full-text extraction silently degraded to abstract-only.
    """

    @pytest.fixture
    def service(self, tmp_path):
        return UploadService(upload_dir=str(tmp_path / "uploads"))

    @pytest.fixture
    def upload_file(self):
        file = MagicMock()
        file.filename = "some_paper.pdf"

        async def _read():
            return b"%PDF-1.4 fake pdf bytes"

        file.read = _read
        return file

    @pytest.fixture
    def request_obj(self):
        from nlp_pillars.schemas import UploadFileRequest

        return UploadFileRequest(
            title="A Retained Paper",
            authors=["Alice"],
            year=2024,
        )

    @pytest.mark.asyncio
    async def test_uploaded_pdf_is_retained_and_url_pdf_resolves(
        self, service, upload_file, request_obj
    ):
        """After a successful upload the stored url_pdf still points at a file.

        Two steps now, because the route saves the bytes and the worker processes the
        path — the request's file handle cannot cross the thread boundary.
        """
        saved_path = await service.save_uploaded_file(upload_file)

        with patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.add_paper', return_value=True), \
             patch.object(UploadService, '_run_full_pipeline') as pipeline:
            pipeline.return_value = PipelineOutcome()

            result = service.run_file_upload_job(
                "nlp-fundamentals", saved_path, upload_file.filename, request_obj
            )

        assert result.paper.url_pdf.startswith("file://")
        retained = Path(result.paper.url_pdf[len("file://"):])
        assert retained.exists(), "uploaded PDF was deleted; url_pdf is now dangling"
        assert retained.read_bytes() == b"%PDF-1.4 fake pdf bytes"

    @pytest.mark.asyncio
    async def test_upload_that_never_reached_the_database_is_cleaned_up(
        self, service, upload_file, request_obj
    ):
        """A file nothing can refer to is still discarded."""
        saved_path = await service.save_uploaded_file(upload_file)

        with patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.add_paper', return_value=False):
            with pytest.raises(UploadError):
                service.run_file_upload_job(
                    "nlp-fundamentals", saved_path, upload_file.filename, request_obj
                )

        assert list(service.upload_dir.glob("*.pdf")) == []

    @pytest.mark.asyncio
    async def test_pipeline_failure_after_insert_still_retains_the_pdf(
        self, service, upload_file, request_obj
    ):
        """Once the papers row exists the file is referenced, so keep it."""
        saved_path = await service.save_uploaded_file(upload_file)

        with patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.add_paper', return_value=True), \
             patch.object(UploadService, '_run_full_pipeline') as pipeline:
            pipeline.side_effect = RuntimeError("summarizer exploded")

            with pytest.raises(RuntimeError):
                service.run_file_upload_job(
                    "nlp-fundamentals", saved_path, upload_file.filename, request_obj
                )

        assert len(list(service.upload_dir.glob("*.pdf"))) == 1

    @pytest.mark.asyncio
    async def test_a_file_that_vanished_before_the_worker_ran_fails_loudly(
        self, service, upload_file, request_obj
    ):
        """The gap between saving the bytes and the job starting is real, and an
        empty upload directory must not surface three stages later as a broken PDF."""
        saved_path = await service.save_uploaded_file(upload_file)
        Path(saved_path).unlink()

        with pytest.raises(UploadError, match="no longer on disk"):
            service.run_file_upload_job(
                "nlp-fundamentals", saved_path, upload_file.filename, request_obj
            )


class TestTitleGuessIsSkippedWhenSomethingAuthoritativeWillOverwriteIt:
    """The URL upload used to parse the PDF twice, and throw the first pass away.

    ``_create_paper_ref_from_url`` extracted the whole PDF to guess a title from its
    first plausible line, and ``_enrich_from_arxiv`` then overwrote that title — plus
    the authors, year, abstract and venue — because for an arXiv paper the API is
    authoritative. ``_run_full_pipeline`` parsed the same file again for the real
    ingest. Measured inside the container on the captain's 4.71 MB PDF: 8.4 seconds
    per upload spent producing a string that was discarded on the next line.

    The fallback is not removed, only made conditional, so the two tests that matter
    are "skipped when arXiv answers" and "still there when it does not".
    """

    @pytest.fixture
    def service(self, tmp_path):
        return UploadService(upload_dir=str(tmp_path / "uploads"))

    @staticmethod
    def _arxiv_answering(title="Attention Is All You Need"):
        result = Mock()
        result.title = title
        author = Mock()
        author.name = "A. Vaswani"
        result.authors = [author]
        result.published = Mock(year=2017)
        result.summary = "The dominant sequence transduction models..."
        result.journal_ref = None
        result.primary_category = "cs.CL"

        client = Mock()
        client.results.return_value = iter([result])
        return client

    def test_an_arxiv_url_does_not_parse_the_pdf_for_a_title(self, service):
        with patch('nlp_pillars.services.paper_metadata.arxiv.Search'), \
             patch('nlp_pillars.services.paper_metadata.arxiv.Client',
                   return_value=self._arxiv_answering()), \
             patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.extract_text') as extract:
            paper = service._create_paper_ref_from_url(
                "https://arxiv.org/pdf/1706.03762", "/tmp/nonexistent.pdf", None, None
            )

        extract.assert_not_called()
        assert paper.title == "Attention Is All You Need"
        assert paper.id == "arxiv:1706.03762"

    def test_a_non_arxiv_url_still_gets_its_title_from_the_pdf(self, service):
        with patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.extract_text',
                   return_value="A Perfectly Good Title From The Paper\nmore body") as extract:
            paper = service._create_paper_ref_from_url(
                "https://example.com/paper.pdf", "/tmp/nonexistent.pdf", None, None
            )

        extract.assert_called_once()
        assert paper.title == "A Perfectly Good Title From The Paper"

    def test_a_failed_arxiv_lookup_falls_back_to_the_pdf(self, service):
        """Otherwise the saving becomes "titled after its hostname, forever"."""
        client = Mock()
        client.results.return_value = iter([])  # StopIteration: not found

        with patch('nlp_pillars.services.paper_metadata.arxiv.Search'), \
             patch('nlp_pillars.services.paper_metadata.arxiv.Client', return_value=client), \
             patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.extract_text',
                   return_value="The Title Only The PDF Knows\nmore body") as extract:
            paper = service._create_paper_ref_from_url(
                "https://arxiv.org/pdf/1706.03762", "/tmp/nonexistent.pdf", None, None
            )

        extract.assert_called_once()
        assert paper.title == "The Title Only The PDF Knows"

    def test_a_title_override_never_parses_the_pdf(self, service):
        with patch.object(UploadService, '_enrich_from_semantic_scholar', lambda self, p: p), \
             patch('nlp_pillars.services.upload_service.extract_text') as extract:
            paper = service._create_paper_ref_from_url(
                "https://example.com/paper.pdf", "/tmp/nonexistent.pdf",
                "The User's Own Title", None,
            )

        extract.assert_not_called()
        assert paper.title == "The User's Own Title"


class TestAFailedPipelineIsNotReportedAsASuccessfulUpload:
    """"uploaded successfully! Triggered: pipeline_error: ..." was the whole bug.

    ``_run_full_pipeline`` wrapped its entire body in ``except Exception``, appended
    the exception to ``actions_triggered`` as a pseudo-action, and the route answered
    ``success=True``. Summarizer, synthesis, quiz and vectors could all fail while the
    upload read as green.
    """

    @pytest.fixture
    def service(self, tmp_path):
        return UploadService(upload_dir=str(tmp_path / "uploads"))

    @pytest.fixture
    def paper(self):
        return PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[], url_pdf="x")

    @staticmethod
    def _parsed(text="body text"):
        """A real ParsedPaper: SummarizerInput validates its fields, so a Mock here
        fails validation and the test ends up exercising the wrong failure."""
        from nlp_pillars.schemas import ParsedPaper

        return ParsedPaper(
            paper_ref=PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[]),
            full_text=text,
            chunks=[text],
        )

    @staticmethod
    def _note():
        from nlp_pillars.schemas import PaperNote

        return PaperNote(
            paper_id="arxiv:1706.03762",
            pillar_id="nlp-fundamentals",
            problem="p",
            method="m",
            findings=["f"],
            limitations=["l"],
            future_work=["w"],
            key_terms=["k"],
        )

    @staticmethod
    def _pillar_config():
        from nlp_pillars.schemas import PillarConfig

        return PillarConfig(
            id="nlp-fundamentals", name="NLP", goal="learn", focus_areas=["a"]
        )

    def test_a_summarizer_failure_is_reported_as_a_failure(self, service, paper):
        with patch.object(service.ingest_agent, 'ingest', return_value=self._parsed()), \
             patch('nlp_pillars.services.upload_service.db') as mock_db, \
             patch('nlp_pillars.services.upload_service.SummarizerAgent') as summarizer, \
             patch('nlp_pillars.services.upload_service.vectors') as mock_vectors:
            mock_db.get_recent_notes.return_value = []
            summarizer.run.side_effect = RuntimeError("model refused")
            mock_vectors.upsert_text.return_value = 12

            outcome = service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals",
                run_summarizer=True, generate_quiz=False,
            )

        assert not outcome.ok
        assert any("summarizer" in e for e in outcome.errors)
        assert "model refused" in " ".join(outcome.errors)
        # The failure is NOT smuggled in as an action.
        assert all("summarizer" != a for a in outcome.actions_triggered)
        assert not any("pipeline_error" in a for a in outcome.actions_triggered)

    def test_the_outcome_says_added_but_not_processed(self, service, paper):
        """The two facts stay apart now that the run row carries them.

        ``UploadJobResult`` reaching the caller at all means the papers row exists —
        the job raises otherwise — and ``outcome.ok`` says whether the follow-on steps
        finished. run_service._finish_upload turns exactly this pair into a failed run
        whose result payload still says ``added: true``.
        """
        outcome = PipelineOutcome(
            actions_triggered=["text_extraction"],
            errors=["summarizer: model refused"],
        )
        assert outcome.ok is False
        assert outcome.actions_triggered == ["text_extraction"]
        # The failure is not smuggled in as a fifth kind of action.
        assert not any("pipeline_error" in a for a in outcome.actions_triggered)

    def test_a_clean_run_still_reads_as_a_clean_success(self, service, paper):
        with patch.object(service.ingest_agent, 'ingest', return_value=self._parsed()), \
             patch('nlp_pillars.services.upload_service.db') as mock_db, \
             patch('nlp_pillars.services.upload_service.SummarizerAgent') as summarizer, \
             patch('nlp_pillars.services.upload_service.SynthesisAgent'), \
             patch('nlp_pillars.services.upload_service.QuizAgent') as quiz, \
             patch.object(UploadService, '_get_pillar_config',
                          return_value=self._pillar_config()), \
             patch('nlp_pillars.services.upload_service.vectors') as mock_vectors:
            mock_db.get_recent_notes.return_value = []
            summarizer.run.return_value = self._note()
            quiz.run.return_value = [Mock()]
            mock_vectors.upsert_text.return_value = 12

            outcome = service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals",
                run_summarizer=True, generate_quiz=True,
            )

        assert outcome.ok, outcome.errors
        assert outcome.actions_triggered == [
            "text_extraction", "summarizer", "lesson_synthesis",
            "quiz_generation", "vector_storage",
        ]

    def test_zero_vectors_from_a_non_empty_paper_is_a_failure(self, service, paper):
        """upsert_text returns 0 for an empty document and a dead Qdrant alike."""
        with patch.object(service.ingest_agent, 'ingest', return_value=self._parsed()), \
             patch('nlp_pillars.services.upload_service.db'), \
             patch('nlp_pillars.services.upload_service.vectors') as mock_vectors:
            mock_vectors.upsert_text.return_value = 0

            outcome = service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals",
                run_summarizer=False, generate_quiz=False,
            )

        assert not outcome.ok
        assert any("vector_storage" in e for e in outcome.errors)

    def test_a_failed_ingest_stops_the_pipeline_but_keeps_the_paper(self, service, paper):
        with patch.object(service.ingest_agent, 'ingest',
                          side_effect=RuntimeError("pdf is garbage")):
            outcome = service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals",
                run_summarizer=True, generate_quiz=True,
            )

        assert not outcome.ok
        assert outcome.actions_triggered == []
        assert any("text_extraction" in e for e in outcome.errors)


class TestUploadDefaultsMatchTheDiscoveryPath:
    """Discovery hardcodes ``enable_quiz=True`` and summarises unconditionally.

    An uploaded paper used to get a bare row unless the user ticked two boxes, and
    ticking only the second produced nothing at all in silence.
    """

    def test_both_post_upload_actions_default_on(self):
        from nlp_pillars.schemas import UploadFileRequest, UploadUrlRequest

        url = UploadUrlRequest(url="https://arxiv.org/pdf/1706.03762")
        assert url.run_summarizer is True
        assert url.generate_quiz is True

        uploaded = UploadFileRequest(title="A Paper")
        assert uploaded.run_summarizer is True
        assert uploaded.generate_quiz is True

    def test_a_quiz_without_a_summarizer_says_so_instead_of_doing_nothing(
        self, tmp_path
    ):
        """The gating is real — the quiz is built from the summarizer's PaperNote —
        but it used to be enforced by the quiz block simply sitting inside
        ``if run_summarizer:``, so the cards never appeared and nothing said why."""
        service = UploadService(upload_dir=str(tmp_path / "uploads"))
        paper = PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[], url_pdf="x")

        parsed = Mock()
        parsed.full_text = "body text"
        parsed.paper_ref = PaperRef(id="arxiv:1706.03762", title="A Paper", authors=[])

        with patch.object(service.ingest_agent, 'ingest', return_value=parsed), \
             patch('nlp_pillars.services.upload_service.db'), \
             patch('nlp_pillars.services.upload_service.QuizAgent') as quiz, \
             patch('nlp_pillars.services.upload_service.vectors') as mock_vectors:
            mock_vectors.upsert_text.return_value = 12

            outcome = service._run_full_pipeline(
                paper=paper, pillar_id="nlp-fundamentals",
                run_summarizer=False, generate_quiz=True,
            )

        quiz.run.assert_not_called()
        assert not outcome.ok
        assert any("quiz_generation" in e and "Run Summarizer" in e
                   for e in outcome.errors)


class TestTheUploadFormSendsWhatTheUserChose:
    """Template-level guards for the two UI halves of the same fix."""

    @pytest.fixture
    def template(self):
        return (
            Path(__file__).resolve().parents[1]
            / "webui" / "templates" / "pillar_detail.html"
        ).read_text()

    def test_all_four_checkboxes_render_checked(self, template):
        for box_id in (
            "file-run-summarizer", "file-generate-quiz",
            "url-run-summarizer", "url-generate-quiz",
        ):
            tag = re.search(rf'<input[^>]*id="{box_id}"[^>]*>', template)
            assert tag, f"{box_id} is gone"
            assert "checked" in tag.group(0), f"{box_id} no longer defaults on"

    def test_the_file_upload_sends_both_flags_explicitly(self, template):
        """An unchecked box is absent from FormData, and the server now defaults these
        to true — so omitting them would flip the user's choice to its opposite."""
        assert "formData.append('run_summarizer'" in template
        assert "formData.append('generate_quiz'" in template

    def test_the_page_no_longer_calls_a_failed_upload_a_success(self, template):
        """No success alert at all now: completion is the progress panel reaching a
        terminal state, and the outcome box distinguishes "added, some steps failed"
        from "nothing was added"."""
        assert "showUploadSuccess" not in template
        assert "showUploadResult" not in template
        assert "renderUploadOutcome" in template
        assert "result.added" in template

    def test_the_upload_panel_follows_a_run_instead_of_awaiting_a_result(
        self, template
    ):
        """The whole point of the change: the routes answer 202 with a run id and the
        page attaches the shared progress component to it."""
        assert "/static/run-progress.js" in template
        assert "RunProgress.attach(runId" in template
        assert "RunProgress.reattach(" in template
        # A run that is not an upload must not be rendered under this heading — the
        # stored run id is shared with the Pipeline and Discovery pages.
        assert "run.kind !== 'upload'" in template

    def test_recent_uploads_reads_the_run_records(self, template):
        """It used to read an in-memory dict that was empty on every page load."""
        assert "/api/pipeline-runs?pillar_id=" in template
        assert "${pillarId}/uploads/recent" not in template

    def test_the_dead_status_endpoints_are_gone(self):
        """Two endpoints served a process-local dict the scheduler could not see and
        no page ever polled. Leaving them would leave two sources of truth."""
        router = (
            Path(__file__).resolve().parents[1]
            / "webui" / "routers" / "api" / "uploads.py"
        ).read_text()
        assert '@router.get' not in router, "the upload router should only POST now"
        assert 'uploads/status/' not in router
        assert 'get_recent_uploads' not in router

        from nlp_pillars import schemas
        from nlp_pillars.services import upload_service as svc

        assert not hasattr(schemas, "UploadStatus")
        assert not hasattr(schemas, "UploadResponse")
        assert not hasattr(svc.UploadService, "get_upload_status")
        assert not hasattr(svc.UploadService, "get_recent_uploads")
