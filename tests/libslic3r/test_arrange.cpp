#include <algorithm>
#include <catch2/catch_all.hpp>
#include "libslic3r/libslic3r.h"
#include "libslic3r/Polygon.hpp"
#include "libslic3r/Point.hpp"
#include <string>
#include "libslic3r/Config.hpp"
#include <vector>
#include <limits>
#include <cstddef>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_message.hpp>
#include "libslic3r/Arrange.hpp"
#include "libslic3r/BoundingBox.hpp"
#include "libslic3r/ClipperUtils.hpp"
#include "libslic3r/ExPolygon.hpp"
#include "libslic3r/Geometry/ConvexHull.hpp"
#include "libslic3r/Line.hpp"
#include "libslic3r/Print.hpp"
#include "libslic3r/PrintConfig.hpp"
#include "libslic3r/MultiMaterialSegmentation.hpp"

using namespace Slic3r;
using namespace Slic3r::arrangement;

namespace {

using Catch::Matchers::WithinRel;

// Square of the given (scaled) side, lower-left at the origin. bed_idx starts at
// 0 because arrange() seeds the nester's bin from it (see ModelArrange.cpp).
ArrangePolygon make_square(coord_t side)
{
    ArrangePolygon ap;
    Polygon        p;
    p.points = {Point(0, 0), Point(side, 0), Point(side, side), Point(0, side)};
    ap.poly  = ExPolygon(p);
    ap.bed_idx = 0;
    return ap;
}

ArrangePolygons squares(int n, double side_mm, double height_mm = 0.)
{
    ArrangePolygons items;
    for (int i = 0; i < n; ++i) {
        items.emplace_back(make_square(scaled(side_mm)));
        items.back().height = height_mm;
    }
    return items;
}

// 64-gon disc of the given diameter, centred on the origin.
ArrangePolygon make_disc(double diameter_mm, double height_mm)
{
    Polygon p;
    for (int i = 0; i < 64; ++i) {
        const double a = 2. * PI * i / 64;
        p.points.emplace_back(scaled(diameter_mm / 2 * std::cos(a)), scaled(diameter_mm / 2 * std::sin(a)));
    }
    ArrangePolygon ap;
    ap.poly    = ExPolygon(p);
    ap.bed_idx = 0;
    ap.height  = height_mm;
    return ap;
}

// Bed [0,0]..[w,h] in scaled coordinates.
BoundingBox bed(double w_mm, double h_mm)
{
    return BoundingBox(Point(0, 0), Point(scaled(w_mm), scaled(h_mm)));
}

// The default progress callback prints to stdout; silence it.
ArrangeParams quiet_params(coord_t min_dist = 0)
{
    ArrangeParams p{min_dist};
    p.progressind = [](unsigned, std::string) {};
    return p;
}

ExPolygons placed_shapes(const ArrangePolygons &items)
{
    ExPolygons out;
    out.reserve(items.size());
    for (const ArrangePolygon &ap : items)
        out.emplace_back(ap.transformed_poly());
    return out;
}

// Area double-counted across the shapes: the sum counts overlaps twice, the
// union once, so the difference is the overlapping area (0 when disjoint).
double overlap_area(const ExPolygons &shapes)
{
    double sum = 0;
    for (const ExPolygon &e : shapes)
        sum += e.area();
    double uni = 0;
    for (const ExPolygon &e : union_ex(shapes))
        uni += e.area();
    return sum - uni;
}

// Relative tolerance absorbs the area-unit rounding the clipper union introduces.
bool disjoint(const ExPolygons &shapes)
{
    double total = 0;
    for (const ExPolygon &e : shapes)
        total += e.area();
    return overlap_area(shapes) <= total * 1e-9;
}

void require_no_overlap(const ArrangePolygons &items)
{
    REQUIRE(disjoint(placed_shapes(items)));
}

// The sequential-print floor is chosen by comparing object height against the nozzle,
// so the two are defined together and every expectation is derived from them.
constexpr double NOZZLE_HEIGHT_MM = 2.5;
constexpr double CLEARANCE_MM     = 30.;
constexpr double NOZZLE_FLOOR_MM  = MAX_OUTER_NOZZLE_DIAMETER;

ArrangeParams seq_print_params(coord_t min_dist)
{
    ArrangeParams p       = quiet_params(min_dist);
    p.is_seq_print        = true;
    p.clearance_radius    = float(CLEARANCE_MM);
    p.nozzle_height       = float(NOZZLE_HEIGHT_MM);
    p.object_skirt_offset = 0.f;
    return p;
}

// update_selected_items_inflation reads the bed out of the config to cap inflation.
DynamicPrintConfig bed_config()
{
    DynamicPrintConfig c;
    c.set_key_value("printable_area", new ConfigOptionPoints{{0, 0}, {200, 0}, {200, 200}, {0, 200}});
    return c;
}

ArrangePolygons squares_of_heights(const std::vector<double> &heights_mm)
{
    ArrangePolygons items;
    for (double height_mm : heights_mm)
        items.push_back(squares(1, 20., height_mm).front());
    return items;
}

// Arranges as the arrange job does for a sequential print, on the bed_config() bed.
void arrange_by_object(ArrangePolygons &items, ArrangePolygons &fixed)
{
    DynamicPrintConfig cfg = bed_config();
    ArrangeParams      p   = seq_print_params(0);
    p.bed_shrink_x         = BED_SHRINK_SEQ_PRINT;
    p.bed_shrink_y         = BED_SHRINK_SEQ_PRINT;
    // Rod and lid above the objects, so they need not each go to a bed of their own.
    p.clearance_height_to_rod = float(NOZZLE_HEIGHT_MM * 10);
    p.clearance_height_to_lid = float(NOZZLE_HEIGHT_MM * 40);
    for (ArrangePolygon &ap : items)
        ap.extrude_ids = {1}; // the sequential sort reads extrude_ids.front()

    update_arrange_params(p, &cfg, items);
    update_selected_items_inflation(items, &cfg, p);
    update_unselected_items_inflation(fixed, &cfg, p);
    arrange(items, fixed, get_shrink_bedpts(&cfg, p), p);
}

} // namespace

// Prove the overlap check the other tests rely on actually detects overlap.
TEST_CASE("overlap_area detects overlap and ignores touching edges", "[Arrange]")
{
    auto square_at = [](double x_mm) {
        ArrangePolygon ap = make_square(scaled(20.));
        ap.translation    = Vec2crd(scaled(x_mm), 0);
        return ap.transformed_poly();
    };
    ExPolygon a = square_at(0.);

    SECTION("disjoint shapes are reported disjoint") {
        REQUIRE(disjoint({a, square_at(30.)}));
    }
    SECTION("edge-touching shapes are reported disjoint") {
        REQUIRE(disjoint({a, square_at(20.)}));
    }
    SECTION("overlapping shapes are not, and the area is measured") {
        REQUIRE_FALSE(disjoint({a, square_at(10.)}));
        REQUIRE_THAT(overlap_area({a, square_at(10.)}),
                     WithinRel(double(scaled(10.)) * scaled(20.), 1e-9)); // 10x20 mm
    }
}

TEST_CASE("Arrange places every item on the physical bed", "[Arrange]")
{
    ArrangePolygons items = squares(5, 20.);
    arrange(items, bed(200, 200), quiet_params(scaled(1.)));

    for (const ArrangePolygon &ap : items)
        REQUIRE(ap.bed_idx == 0);
}

TEST_CASE("Arranged items stay within the bed", "[Arrange]")
{
    ArrangePolygons items = squares(6, 30.);
    arrange(items, bed(200, 200), quiet_params(scaled(1.)));

    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        REQUIRE(bed(200, 200).contains(ap.transformed_poly().contour.bounding_box()));
    }
}

TEST_CASE("Arranged items do not overlap", "[Arrange]")
{
    ArrangePolygons items = squares(6, 40.);
    arrange(items, bed(250, 250), quiet_params(scaled(2.)));

    require_no_overlap(items);
}

TEST_CASE("Arrange spaces items by their inflation", "[Arrange]")
{
    // Per-item inflation is how the arranger enforces clearance (the GUI fills it
    // from min_obj_distance). Two items inflated 4mm each end up >= 8mm apart.
    ArrangePolygons items = squares(4, 20.);
    for (ArrangePolygon &ap : items)
        ap.inflation = scaled(4.);
    arrange(items, bed(200, 200), quiet_params());

    // Axis-aligned squares are their own bounding boxes, so the clearance between
    // a pair is the distance between their boxes (1mm slack for nester rounding).
    std::vector<BoundingBox> boxes;
    for (const ExPolygon &e : placed_shapes(items))
        boxes.push_back(e.contour.bounding_box());

    double min_gap = std::numeric_limits<double>::max();
    for (size_t i = 0; i < boxes.size(); ++i)
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            coord_t sx = std::max<coord_t>(0, std::max(boxes[j].min.x() - boxes[i].max.x(),
                                                       boxes[i].min.x() - boxes[j].max.x()));
            coord_t sy = std::max<coord_t>(0, std::max(boxes[j].min.y() - boxes[i].max.y(),
                                                       boxes[i].min.y() - boxes[j].max.y()));
            min_gap = std::min(min_gap, std::sqrt(double(sx) * sx + double(sy) * sy));
        }

    REQUIRE(min_gap >= double(scaled(8.)) - double(scaled(0.5)));
}

TEST_CASE("An item larger than the bed cannot be placed", "[Arrange]")
{
    ArrangePolygons items;
    items.emplace_back(make_square(scaled(20.)));
    items.emplace_back(make_square(scaled(400.))); // far bigger than the bed
    // An item that cannot be placed keeps the position it came in with, so the
    // caller can leave it where the user had it.
    items[1].translation = {scaled(10.), scaled(20.)};

    arrange(items, bed(200, 200), quiet_params(scaled(1.)));

    REQUIRE(items[0].bed_idx == 0);
    REQUIRE(items[1].bed_idx == UNARRANGED);
    REQUIRE(items[1].translation.x() == scaled(10.));
    REQUIRE(items[1].translation.y() == scaled(20.));
}

TEST_CASE("An item filling the printable area is arranged", "[Arrange]")
{
    // The arrange bed keeps a margin from the printable area and the item is widened by
    // its brim. Neither may keep an item that fits the printable area from being placed.
    DynamicPrintConfig cfg   = bed_config(); // 200 x 200
    ArrangeParams      p     = quiet_params();

    ArrangePolygons items = squares(1, 197.);
    items.front().brim_width = 1.;
    update_arrange_params(p, &cfg, items);
    update_selected_items_inflation(items, &cfg, p);
    arrange(items, get_shrink_bedpts(&cfg, p), p);
    REQUIRE(items.front().bed_idx == 0);

    // Larger than the printable area is still refused.
    ArrangeParams   p_large = quiet_params();
    ArrangePolygons large   = squares(1, 201.);
    update_arrange_params(p_large, &cfg, large);
    update_selected_items_inflation(large, &cfg, p_large);
    arrange(large, get_shrink_bedpts(&cfg, p_large), p_large);
    REQUIRE(large.front().bed_idx == UNARRANGED);
}

TEST_CASE("Arranging uses the whole bed when the extruders share no printable area", "[Arrange]")
{
    // Two half-bed areas that only touch. An item wider than either half still fits the bed.
    DynamicPrintConfig cfg = bed_config(); // 200 x 200
    cfg.set_key_value("extruder_printable_area", new ConfigOptionPointsGroups{
        { { 0., 0. }, { 100., 0. }, { 100., 200. }, { 0., 200. } },
        { { 100., 0. }, { 200., 0. }, { 200., 200. }, { 100., 200. } } });
    ArrangeParams p = quiet_params();

    ArrangePolygons items = squares(1, 150.);
    update_arrange_params(p, &cfg, items);
    update_selected_items_inflation(items, &cfg, p);
    const Points bedpts = get_shrink_bedpts(&cfg, p);
    // The arrange jobs read the four corners.
    REQUIRE(bedpts.size() == 4);
    arrange(items, bedpts, p);
    CHECK(items.front().bed_idx == 0);

    // The bed still bounds the arrangement.
    ArrangeParams   p_large = quiet_params();
    ArrangePolygons large   = squares(1, 201.);
    update_arrange_params(p_large, &cfg, large);
    update_selected_items_inflation(large, &cfg, p_large);
    arrange(large, get_shrink_bedpts(&cfg, p_large), p_large);
    CHECK(large.front().bed_idx == UNARRANGED);
}

TEST_CASE("Items overflowing one bed spill onto virtual beds", "[Arrange]")
{
    ArrangePolygons items = squares(8, 90.); // eight 90mm squares cannot share a 200x200 bed
    arrange(items, bed(200, 200), quiet_params(scaled(2.)));

    int max_bed = 0;
    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx >= 0); // placed somewhere
        max_bed = std::max(max_bed, ap.bed_idx);
    }
    REQUIRE(max_bed >= 1); // at least one on a virtual bed
}

TEST_CASE("Arrange handles an empty input", "[Arrange]")
{
    ArrangePolygons items;
    REQUIRE_NOTHROW(arrange(items, bed(200, 200), quiet_params()));
    REQUIRE(items.empty());
}

TEST_CASE("Arrange without final alignment keeps items disjoint", "[Arrange]")
{
    // do_final_align = false selects Alignment::DONT_ALIGN (skips recentering).
    ArrangePolygons items  = squares(6, 40.);
    ArrangeParams   params = quiet_params(scaled(2.));
    params.do_final_align  = false;

    arrange(items, bed(250, 250), params);

    for (const ArrangePolygon &ap : items)
        REQUIRE(ap.bed_idx == 0);
    require_no_overlap(items);
}

TEST_CASE("Arrange centers a pile that fits on the custom alignment point", "[Arrange]")
{
    ArrangePolygons items  = squares(4, 30.);
    ArrangeParams   params = quiet_params(scaled(2.));
    params.align_center    = Vec2d(0.3, 0.7);

    arrange(items, bed(250, 250), params);

    BoundingBox pile;
    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        pile.merge(ap.transformed_poly().contour.bounding_box());
    }
    // best_object_pos is align_center scaled to the bed.
    const Point expected(scaled(250. * 0.3), scaled(250. * 0.7));
    REQUIRE(std::abs(pile.center().x() - expected.x()) <= scaled(0.5));
    REQUIRE(std::abs(pile.center().y() - expected.y()) <= scaled(0.5));
    require_no_overlap(items);
}

// A row of four 40 mm parts centred on the 30% mark of a 200 x 50 mm bed would start 24 mm
// left of it (#15727). The pile stops at the left edge instead.
TEST_CASE("Arrange keeps a custom-aligned pile within the bed", "[Arrange]")
{
    const BoundingBox bed_   = bed(200, 50);
    ArrangePolygons   items  = squares(4, 40.);
    ArrangeParams     params = quiet_params(scaled(2.));
    params.align_center      = Vec2d(0.3, 0.5);

    arrange(items, bed_, params);

    coord_t leftmost = std::numeric_limits<coord_t>::max();
    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        const BoundingBox bb = ap.transformed_poly().contour.bounding_box();
        CHECK(bed_.contains(bb));
        leftmost = std::min(leftmost, bb.min.x());
    }
    // Stopped at the edge it was aimed at, not re-centred.
    CHECK(leftmost < scaled(10.));
    require_no_overlap(items);
}

// Arranging a selection leaves the unselected parts where they are, but the final alignment
// centres the box around both, so the selection does not move by its own centring. The check
// against the bed and the fixed part must test where the selection actually lands.
TEST_CASE("Arrange keeps a selection on the bed and clear of a fixed part", "[Arrange]")
{
    ArrangePolygons fixed = squares(1, 40.);
    fixed.front().translation = {scaled(80.), scaled(80.)};
    ArrangePolygons items = squares(3, 40.);
    for (ArrangePolygon &ap : items)
        ap.inflation = scaled(1.);

    arrange(items, fixed, bed(200, 200), quiet_params());

    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        REQUIRE(bed(200, 200).contains(ap.transformed_poly().contour.bounding_box()));
    }
    items.insert(items.end(), fixed.begin(), fixed.end());
    require_no_overlap(items);
}

// A belt printer starts its parts at the leading end of the belt (best_object_pos 0.5, 0.05).
// Centring a pile on a point that close to the edge pushed everything longer than the room
// around it off the bed: four 90 mm parts on a 95 x 500 mm belt ended with one across the
// edge and one outside, with 290 mm of belt free behind them. The pile stops at the edge.
TEST_CASE("Arrange keeps a pile aligned near an edge on the bed", "[Arrange][belt]")
{
    const BoundingBox belt   = bed(95, 500);
    ArrangePolygons   items  = squares(4, 90.);
    ArrangeParams     params = quiet_params(scaled(2.));
    params.align_center      = Vec2d(0.5, 0.05);
    params.is_belt           = true;
    params.belt_axis         = 1;
    params.belt_tilt_slope   = 1.f;

    arrange(items, belt, params);

    coord_t lowest = std::numeric_limits<coord_t>::max();
    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        const BoundingBox bb = ap.transformed_poly().contour.bounding_box();
        CHECK(belt.contains(bb));
        lowest = std::min(lowest, bb.min.y());
    }
    // Snapped to the edge it was aimed at, less the spacing margin, not re-centred.
    CHECK(lowest < scaled(10.));
    require_no_overlap(items);
}

// Flat-bed printers with an off-centre best_object_pos (the A1 mini and the H2 family) get
// the same clamp: centring the pile on that point used to push it off the bed (#15727).
TEST_CASE("Arrange keeps a flat-bed pile aligned near an edge on the bed", "[Arrange]")
{
    const BoundingBox bed_   = bed(95, 500);
    ArrangePolygons   items  = squares(4, 90.);
    ArrangeParams     params = quiet_params(scaled(2.));
    params.align_center      = Vec2d(0.5, 0.05);

    arrange(items, bed_, params);

    coord_t lowest = std::numeric_limits<coord_t>::max();
    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        const BoundingBox bb = ap.transformed_poly().contour.bounding_box();
        CHECK(bed_.contains(bb));
        lowest = std::min(lowest, bb.min.y());
    }
    // Stopped at the edge it was aimed at, not re-centred.
    CHECK(lowest < scaled(10.));
    require_no_overlap(items);
}

// On a belt the parts print in belt order, so two colours that alternate along the
// belt, or sit side by side, cost a filament change on every shared layer. Arrange
// keeps each colour together: no part shares belt length with a part of another
// colour, counting the tilted layers that run cot(angle) * height past its far edge,
// whichever end of the belt prints first.
TEST_CASE("Arrange groups the colours of a belt print along the belt", "[Arrange][belt]")
{
    const bool reversed = GENERATE(false, true);
    CAPTURE(reversed);
    const BoundingBox belt   = bed(95, 500);
    ArrangePolygons   items  = squares(6, 30., 20.);
    for (size_t i = 0; i < items.size(); ++i)
        items[i].extrude_ids = { int(i % 3) + 1 };   // three colours, two parts each
    ArrangeParams params     = quiet_params(scaled(2.));
    params.align_center      = Vec2d(0.5, 0.05);
    params.is_belt           = true;
    params.belt_axis         = 1;
    params.belt_reversed     = reversed;
    params.belt_tilt_slope   = 1.f;   // 45 degrees

    arrange(items, belt, params);
    require_no_overlap(items);

    // Belt position in print order, so the same check serves both directions.
    const coord_t dir = reversed ? -1 : 1;
    auto start = [&](const ArrangePolygon &ap) { const BoundingBox bb = ap.transformed_poly().contour.bounding_box(); return dir * (reversed ? bb.max.y() : bb.min.y()); };
    auto end   = [&](const ArrangePolygon &ap) { const BoundingBox bb = ap.transformed_poly().contour.bounding_box(); return dir * (reversed ? bb.min.y() : bb.max.y()) + scaled(ap.height * params.belt_tilt_slope); };

    for (const ArrangePolygon &ap : items) {
        REQUIRE(ap.bed_idx == 0);
        CHECK(belt.contains(ap.transformed_poly().contour.bounding_box()));
    }
    for (const ArrangePolygon &a : items)
        for (const ArrangePolygon &b : items) {
            if (a.extrude_ids == b.extrude_ids)
                continue;
            // The part printed later starts after the earlier one has finished.
            const coord_t earlier_end = start(a) <= start(b) ? end(a) : end(b);
            const coord_t later_start = std::max(start(a), start(b));
            INFO("colour " << a.extrude_ids.front() << " vs " << b.extrude_ids.front());
            CHECK(earlier_end <= later_start);
        }
}

TEST_CASE("Sequential print floors the object distance by object height", "[Arrange]")
{
    // The only place sequential-print clearance is enforced. The arrange menu offers
    // no floor of its own, so a stored 0 has to be raised here or not at all.
    struct Case
    {
        std::string         description;
        std::vector<double> heights;
        double              skirt_offset_mm;
        double              expected_floor_mm;
    };

    auto c = GENERATE(values<Case>({
        {"objects taller than the nozzle need the full clearance",     {NOZZLE_HEIGHT_MM * 2, NOZZLE_HEIGHT_MM * 2}, 0., CLEARANCE_MM},
        {"an object exactly at the nozzle height counts as tall",      {NOZZLE_HEIGHT_MM,     NOZZLE_HEIGHT_MM},     0., CLEARANCE_MM},
        {"one tall object among short ones is enough",                 {NOZZLE_HEIGHT_MM / 2, NOZZLE_HEIGHT_MM * 2}, 0., CLEARANCE_MM},
        {"objects the nozzle clears keep only the nozzle-width floor", {NOZZLE_HEIGHT_MM / 2, NOZZLE_HEIGHT_MM / 2}, 0., NOZZLE_FLOOR_MM},
        {"a wide skirt raises the floor for short objects",            {NOZZLE_HEIGHT_MM / 2, NOZZLE_HEIGHT_MM / 2}, 3., 6.},
    }));

    DYNAMIC_SECTION(c.description)
    {
        ArrangePolygons    items = squares_of_heights(c.heights);
        DynamicPrintConfig cfg   = bed_config();
        ArrangeParams      p     = seq_print_params(0);
        p.object_skirt_offset    = float(c.skirt_offset_mm);

        update_selected_items_inflation(items, &cfg, p);

        CHECK(p.min_obj_distance >= scaled(c.expected_floor_mm));
        CHECK(p.min_obj_distance <= scaled(c.expected_floor_mm + 0.01));
        // Half each, so a pair ends up a full min_obj_distance apart.
        CHECK(items.front().inflation == p.min_obj_distance / 2);
    }
}

TEST_CASE("Sequential print keeps an object distance already above the floor", "[Arrange]")
{
    const coord_t      stored = scaled(CLEARANCE_MM * 2);
    ArrangePolygons    items  = squares_of_heights({NOZZLE_HEIGHT_MM * 2, NOZZLE_HEIGHT_MM * 2});
    DynamicPrintConfig cfg    = bed_config();
    ArrangeParams      p      = seq_print_params(stored);

    update_selected_items_inflation(items, &cfg, p);
    CHECK(p.min_obj_distance == stored);
}

TEST_CASE("Layered printing does not floor the object distance", "[Arrange]")
{
    ArrangePolygons    items = squares_of_heights({NOZZLE_HEIGHT_MM * 2, NOZZLE_HEIGHT_MM * 2});
    DynamicPrintConfig cfg   = bed_config();
    ArrangeParams      p     = seq_print_params(0);
    p.is_seq_print           = false;

    update_selected_items_inflation(items, &cfg, p);
    CHECK(p.min_obj_distance == 0);
}

// Short objects only need a nozzle's width between them, far less than the clearance
// radius, so the bed may only grow by half that width. A row of four 50 mm objects and
// their gaps is wider than the 200 mm bed but would fit one grown by half the clearance
// radius. Twenty of them overfill the bed, so the pile reaches its edges.
TEST_CASE("Sequential print keeps short objects on the bed and a nozzle apart", "[Arrange]")
{
    ArrangePolygons items = squares(20, 50., NOZZLE_HEIGHT_MM / 2);
    ArrangePolygons fixed;
    arrange_by_object(items, fixed);

    // Grown as Print::sequential_print_clearance_valid() grows them.
    ExPolygons grown;
    for (const ArrangePolygon &ap : items) {
        if (ap.bed_idx != 0)
            continue;
        CHECK(bed(200, 200).contains(ap.transformed_poly().contour.bounding_box()));
        append(grown, offset_ex(ap.transformed_poly(), float(scale_(0.5 * MAX_OUTER_NOZZLE_DIAMETER - 0.1))));
    }
    REQUIRE(grown.size() > 1);
    REQUIRE(disjoint(grown));
}

// Only the disc has to fit the bed; the half gap it is inflated by may reach past the
// edge. Within the bed margin of filling the bed, the arrange bed is cut to the disc's
// size, and rounding must not leave it a hair too small.
TEST_CASE("Sequential print arranges a short object filling the printable area", "[Arrange]")
{
    ArrangePolygons items{make_disc(196., NOZZLE_HEIGHT_MM / 2)};
    ArrangePolygons fixed;
    arrange_by_object(items, fixed);

    REQUIRE(items.front().bed_idx == 0);
    REQUIRE(bed(200, 200).contains(items.front().transformed_poly().contour.bounding_box()));
}

// The largest object and its clearance are wider than the bed less its margins, though
// the object itself fits within them, so every object keeps the margin.
TEST_CASE("Sequential print keeps the bed margin beside a large tall object", "[Arrange]")
{
    ArrangePolygons items = squares(3, 10., NOZZLE_HEIGHT_MM * 2);
    items.push_back(make_disc(170., NOZZLE_HEIGHT_MM * 2));
    ArrangePolygons fixed;
    arrange_by_object(items, fixed);

    const coord_t     margin = scaled(BED_SHRINK_SEQ_PRINT - 0.01); // 0.01 mm for rounding
    const BoundingBox inner(Point(margin, margin), Point(scaled(200.) - margin, scaled(200.) - margin));
    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0)
            CHECK(inner.contains(ap.transformed_poly().contour.bounding_box()));
}

// Short objects keep off an extrusion calibration region. One along the back edge catches
// the region being moved back by a positive bed shrink, which lets objects onto it.
TEST_CASE("Sequential print keeps short objects off a calibration region", "[Arrange]")
{
    const BoundingBox region_bb(Point(0, scaled(185.)), Point(scaled(200.), scaled(200.)));
    ArrangePolygon    region;
    region.poly                     = ExPolygon(region_bb.polygon());
    region.bed_idx                  = 0;
    region.is_virt_object           = true;
    region.is_extrusion_cali_object = true;
    ArrangePolygons items = squares(20, 40., NOZZLE_HEIGHT_MM / 2);
    ArrangePolygons fixed{region};
    arrange_by_object(items, fixed);

    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0)
            CHECK(ap.transformed_poly().contour.bounding_box().max.y() <= region_bb.min.y());
}

// The nester packs an object turned and inflated, and the inflated corners of a turned square
// reach past its inflation, so cutting the arrange bed to the object's size must measure that
// outline. The square is 196 mm wide turned 30 degrees.
TEST_CASE("Sequential print arranges a turned short object filling the printable area", "[Arrange]")
{
    ArrangePolygon item = make_square(scaled(196. / (std::cos(PI / 6) + std::sin(PI / 6))));
    item.height   = NOZZLE_HEIGHT_MM / 2;
    item.rotation = PI / 6;
    ArrangePolygons items{item};
    ArrangePolygons fixed;
    arrange_by_object(items, fixed);

    REQUIRE(items.front().bed_idx == 0);
    REQUIRE(bed(200, 200).contains(items.front().transformed_poly().contour.bounding_box()));
}

// An object closer to the bed edge than the bed margin, as one placed by hand, still takes its
// room when Fill bed adds objects around it, though its inflation crosses the arrange bed's edge.
TEST_CASE("Sequential print keeps short objects off one at the bed edge", "[Arrange]")
{
    ArrangePolygon obstacle = make_square(scaled(40.));
    obstacle.translation    = {scaled(1.), scaled(80.)};
    obstacle.height         = NOZZLE_HEIGHT_MM / 2;
    obstacle.extrude_ids    = {1};
    ArrangePolygons fixed{obstacle};
    ArrangePolygons items = squares(20, 40., NOZZLE_HEIGHT_MM / 2);
    arrange_by_object(items, fixed);

    ExPolygons placed{obstacle.transformed_poly()};
    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0)
            placed.emplace_back(ap.transformed_poly());
    REQUIRE(placed.size() > 1);
    REQUIRE(disjoint(placed));
}

TEST_CASE("Bed exclusions apply by physical nozzle and overlapping Z range", "[Arrange][ExclusionVolume][MultiNozzle]")
{
    ArrangePolygon object = make_square(scaled(20.));
    object.has_z_range = true;
    object.z_min = 0.0;
    object.z_max = 20.0;
    object.bed_exclusion_extruder_ids = {1};

    ArrangePolygon exclusion = make_square(scaled(20.));
    exclusion.is_bed_exclusion = true;
    exclusion.has_z_range = true;
    exclusion.z_min = 10.0;
    exclusion.z_max = 30.0;
    exclusion.bed_exclusion_extruder_id = 1;

    CHECK(bed_exclusion_applies(object, exclusion));

    exclusion.bed_exclusion_extruder_id = 0;
    CHECK_FALSE(bed_exclusion_applies(object, exclusion));

    // Empty physical-nozzle assignments mean unresolved, so arrangement stays conservative.
    object.bed_exclusion_extruder_ids.clear();
    CHECK(bed_exclusion_applies(object, exclusion));

    object.bed_exclusion_extruder_ids = {0};
    object.z_max = 9.0;
    CHECK_FALSE(bed_exclusion_applies(object, exclusion));

    // Touching at one Z plane is intentionally treated as a collision.
    object.z_max = 10.0;
    CHECK(bed_exclusion_applies(object, exclusion));
}

TEST_CASE("Shared bed exclusions apply to every arranged object", "[Arrange][ExclusionVolume]")
{
    ArrangePolygon object = make_square(scaled(20.));
    object.bed_exclusion_extruder_ids = {1};

    ArrangePolygon shared = make_square(scaled(20.));
    shared.is_bed_exclusion = true;
    shared.bed_exclusion_extruder_id = -1;

    CHECK(has_bed_exclusion_regions({shared}));
    CHECK_FALSE(has_bed_exclusion_regions({object}));
    CHECK(bed_exclusion_applies(object, shared));
}

TEST_CASE("Arrange keeps relevant objects outside conditional exclusion obstacles", "[Arrange][ExclusionVolume]")
{
    ArrangePolygon exclusion = make_square(scaled(30.));
    exclusion.translation = Vec2crd(scaled(35.0), scaled(35.0));
    exclusion.is_bed_exclusion = true;
    exclusion.bed_exclusion_extruder_id = 0;

    ArrangePolygons items = squares(4, 20.0);
    for (ArrangePolygon &item : items)
        item.bed_exclusion_extruder_ids = {0};

    arrange(items, {exclusion}, bed(100.0, 100.0), quiet_params(scaled(1.0)));

    const ExPolygon excluded_shape = exclusion.transformed_poly();
    for (const ArrangePolygon &item : items) {
        REQUIRE(item.bed_idx == 0);
        CHECK(intersection(ExPolygons{item.transformed_poly()}, ExPolygons{excluded_shape}).empty());
    }
    require_no_overlap(items);
}

// Geometry levels. Plates of generated shapes, each with its real outline and its convex
// hull as the arrange polygon, on a 256 mm bed with 2 mm spacing.
namespace {

constexpr double BED_MM     = 256.;
constexpr double SPACING_MM = 2.;
// Slack for the gap check: the outline simplification and integer rounding.
constexpr double GAP_SLACK_MM = 0.05;

using Level = ArrangeGeometry;

const char *level_name(Level l) { return arrange_geometry_name(l); }

Polygon circle(double r_mm)
{
    Polygon p;
    for (int i = 0; i < 64; ++i) {
        const double a = 2. * PI * i / 64;
        p.points.emplace_back(scaled(r_mm * std::cos(a)), scaled(r_mm * std::sin(a)));
    }
    return p;
}

Polygon polygon_mm(std::initializer_list<std::pair<double, double>> pts)
{
    Polygon p;
    for (auto [x, y] : pts)
        p.points.emplace_back(scaled(x), scaled(y));
    return p;
}

// L bracket: two arms of length w and thickness t.
ExPolygon shape_L(double w, double t) { return ExPolygon(polygon_mm({{0, 0}, {w, 0}, {w, t}, {t, t}, {t, w}, {0, w}})); }

// C shape opening towards +X.
ExPolygon shape_C(double w, double h, double t)
{
    return ExPolygon(polygon_mm({{0, 0}, {w, 0}, {w, t}, {t, t}, {t, h - t}, {w, h - t}, {w, h}, {0, h}}));
}

ExPolygon shape_disc(double r) { return ExPolygon(circle(r)); }

ExPolygon shape_ring(double R, double r)
{
    ExPolygon ring(circle(R));
    Polygon   hole = circle(r);
    hole.reverse();
    ring.holes.emplace_back(std::move(hole));
    return ring;
}

ArrangePolygon make_item(const ExPolygon &outline)
{
    ArrangePolygon ap;
    ap.outline   = {outline};
    ap.poly      = ExPolygon(Geometry::convex_hull(outline.contour.points));
    ap.inflation = scaled(SPACING_MM / 2.);
    ap.bed_idx   = 0;
    ap.height    = 10.;
    return ap;
}

struct Scenario
{
    std::string            name;
    std::vector<ExPolygon> shapes;
};

void add(Scenario &s, int n, const ExPolygon &shape) { s.shapes.insert(s.shapes.end(), n, shape); }

Scenario scenario(const std::string &name)
{
    Scenario s{name, {}};
    const ExPolygon L = shape_L(60, 15), C = shape_C(50, 40, 10), cyl = shape_disc(12), disc = shape_disc(15);
    if (name == "cyl30") add(s, 30, cyl);
    else if (name == "cap-cyl") add(s, 110, cyl);
    else if (name == "L16") add(s, 16, L);
    else if (name == "cap-L") add(s, 40, L);
    else if (name == "C20") add(s, 20, C);
    else if (name == "mixed24") { add(s, 8, L); add(s, 8, C); add(s, 8, disc); }
    else if (name == "cap-mixed") { add(s, 24, L); add(s, 24, C); add(s, 24, disc); }
    else if (name == "rings") { add(s, 3, shape_ring(40, 30)); add(s, 12, shape_disc(8)); }
    else if (name == "rings-big") { add(s, 6, shape_ring(40, 30)); add(s, 30, shape_disc(8)); }
    else throw std::runtime_error("unknown scenario " + name);
    return s;
}

const std::vector<std::string> ALL_SCENARIOS = {"cyl30", "cap-cyl", "L16", "cap-L", "C20", "mixed24", "cap-mixed", "rings", "rings-big"};

ArrangePolygons scenario_items(const std::string &name)
{
    ArrangePolygons items;
    for (const ExPolygon &e : scenario(name).shapes)
        items.emplace_back(make_item(e));
    return items;
}

ArrangeParams level_params(Level level, bool rotations)
{
    ArrangeParams p   = quiet_params();
    p.allow_rotations = rotations;
    p.geometry        = level;
    return p;
}

struct Run
{
    ArrangePolygons items;
    double          seconds = 0.;
};

Run run_arrange(const std::string &name, Level level, bool rotations)
{
    Run        r{scenario_items(name)};
    const auto t0 = std::chrono::steady_clock::now();
    arrange(r.items, bed(BED_MM, BED_MM), level_params(level, rotations));
    r.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return r;
}

// Arranges the plate with a stop condition that never stops, and returns how often it was
// asked. It is asked once per rotation an item is tried at.
int stop_polls(ArrangePolygons &items, Level level, bool rotations)
{
    ArrangeParams p = level_params(level, rotations);
    int           polls = 0;
    p.stopcondition     = [&polls] { ++polls; return false; };
    arrange(items, bed(BED_MM, BED_MM), p);
    return polls;
}

ExPolygons placed_outline(const ArrangePolygon &ap)
{
    ExPolygons out = ap.outline;
    for (ExPolygon &e : out) {
        e.rotate(ap.rotation);
        e.translate(ap.translation.x(), ap.translation.y());
    }
    return out;
}

// Every pair on one bed must stay at least spacing - slack apart. Growing both outlines by
// half of that with round joins makes them touch exactly at that distance; miter joins would
// report false violations at diagonal corners.
struct Violations
{
    int         overlaps = 0, gaps = 0, outside = 0;
    std::string first;
};

Violations check_layout(const ArrangePolygons &items)
{
    Violations v;
    std::vector<ExPolygons>  placed, grown;
    std::vector<BoundingBox> bbs;
    const float              half_gap = float(scaled((SPACING_MM - GAP_SLACK_MM) / 2.));
    for (const ArrangePolygon &ap : items) {
        placed.emplace_back(placed_outline(ap));
        grown.emplace_back(offset_ex(placed.back(), half_gap, jtRound));
        bbs.emplace_back(get_extents(grown.back()));
    }
    const double      tiny = scaled(0.01) * scaled(0.01);
    const BoundingBox bed_area(Point(-scaled(0.01), -scaled(0.01)), Point(scaled(BED_MM + 0.01), scaled(BED_MM + 0.01)));
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].bed_idx == 0 && !bed_area.contains(get_extents(placed[i]))) {
            ++v.outside;
            if (v.first.empty()) v.first = "item " + std::to_string(i) + " outside bed";
        }
        for (size_t j = i + 1; j < items.size(); ++j) {
            if (items[i].bed_idx != items[j].bed_idx || items[i].bed_idx < 0 || !bbs[i].overlap(bbs[j]))
                continue;
            if (area(intersection_ex(placed[i], placed[j])) > tiny) {
                ++v.overlaps;
                if (v.first.empty()) v.first = "items " + std::to_string(i) + "," + std::to_string(j) + " overlap";
            } else if (area(intersection_ex(grown[i], grown[j])) > tiny) {
                ++v.gaps;
                if (v.first.empty()) v.first = "items " + std::to_string(i) + "," + std::to_string(j) + " closer than spacing";
            }
        }
    }
    return v;
}

void require_valid_layout(const ArrangePolygons &items)
{
    const Violations v = check_layout(items);
    INFO(v.first);
    CHECK(v.overlaps == 0);
    CHECK(v.gaps == 0);
    CHECK(v.outside == 0);
}

struct Metrics
{
    int    on_bed0 = 0;
    double pile_w = 0, pile_h = 0;
};

Metrics metrics(const ArrangePolygons &items)
{
    Metrics     m;
    BoundingBox bb;
    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0) {
            ++m.on_bed0;
            bb.merge(get_extents(placed_outline(ap)));
        }
    if (bb.defined) {
        m.pile_w = unscaled(bb.size().x());
        m.pile_h = unscaled(bb.size().y());
    }
    return m;
}

// Area of the convex hull of everything on bed 0.
double pile_hull_area_mm2(const ArrangePolygons &items)
{
    Points pts;
    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0)
            append(pts, to_points(placed_outline(ap)));
    return std::abs(Geometry::convex_hull(pts).area()) * SCALING_FACTOR * SCALING_FACTOR;
}

// How good a whole arrangement is: fewer items left unplaced, then fewer beds, then more
// items on bed 0, then a smaller pile hull on bed 0.
std::tuple<int, int, int, double> arrangement_cost(const ArrangePolygons &items)
{
    int unplaced = 0, beds = 0;
    for (const ArrangePolygon &ap : items) {
        if (ap.bed_idx < 0) ++unplaced;
        else beds = std::max(beds, ap.bed_idx + 1);
    }
    return {unplaced, beds, -metrics(items).on_bed0, pile_hull_area_mm2(items)};
}

// Discs lying entirely inside the hole of a placed ring.
int discs_in_ring_holes(const ArrangePolygons &items)
{
    Polygons holes;
    for (const ArrangePolygon &ap : items)
        if (ap.bed_idx == 0)
            for (const ExPolygon &e : placed_outline(ap))
                holes.insert(holes.end(), e.holes.begin(), e.holes.end());
    const auto in_a_hole = [&holes](const Polygon &disc) {
        return std::any_of(holes.begin(), holes.end(), [&disc](const Polygon &h) { return diff(Polygons{disc}, Polygons{h}).empty(); });
    };
    return int(std::count_if(items.begin(), items.end(), [&](const ArrangePolygon &ap) {
        return ap.bed_idx == 0 && ap.outline.front().holes.empty() && in_a_hole(placed_outline(ap).front().contour);
    }));
}

// Distance between two disjoint outlines: the closest vertex of one to an edge of the other.
double outline_distance(const ExPolygons &a, const ExPolygons &b)
{
    double     best  = std::numeric_limits<double>::max();
    const auto sweep = [&best](const ExPolygons &from, const ExPolygons &to) {
        const Lines edges = to_lines(to);
        for (const Point &p : to_points(from))
            for (const Line &l : edges) best = std::min(best, l.distance_to(p));
    };
    sweep(a, b);
    sweep(b, a);
    return unscaled(best);
}

// The gap between the two closest items on bed 0.
double closest_gap_mm(const ArrangePolygons &items)
{
    double best = std::numeric_limits<double>::max();
    for (size_t i = 0; i < items.size(); ++i)
        for (size_t j = i + 1; j < items.size(); ++j)
            if (items[i].bed_idx == 0 && items[j].bed_idx == 0)
                best = std::min(best, outline_distance(placed_outline(items[i]), placed_outline(items[j])));
    return best;
}

// Items on bed 0 whose outline reaches into another item's pocket (its convex hull minus
// its outline) by at least min_area_mm2.
int interlocked_items(const ArrangePolygons &items, double min_area_mm2)
{
    std::vector<ExPolygons> outline, pocket;
    for (const ArrangePolygon &ap : items) {
        outline.emplace_back(ap.bed_idx == 0 ? placed_outline(ap) : ExPolygons{});
        pocket.emplace_back(diff_ex(ExPolygons{ExPolygon(Geometry::convex_hull(to_points(outline.back())))}, outline.back()));
    }
    int n = 0;
    for (size_t i = 0; i < items.size(); ++i)
        for (size_t j = 0; j < items.size(); ++j)
            if (i != j && Slic3r::area(intersection_ex(outline[i], pocket[j])) >= scaled(1.) * scaled(1.) * min_area_mm2) {
                ++n;
                break;
            }
    return n;
}

std::string golden_path() { return std::string(TEST_DATA_DIR) + "/arrange_fast_golden.txt"; }

std::string fast_layout_dump(const std::string &name, bool rot, const ArrangePolygons &items)
{
    std::ostringstream os;
    for (size_t i = 0; i < items.size(); ++i)
        os << name << ' ' << int(rot) << ' ' << i << ' ' << items[i].bed_idx << ' ' << items[i].translation.x() << ' '
           << items[i].translation.y() << ' ' << std::llround(items[i].rotation * 1e9) << '\n';
    return os.str();
}

std::vector<std::string> lines_of(const std::string &s, const std::string &prefix)
{
    std::vector<std::string> out;
    std::istringstream       is(s);
    for (std::string l; std::getline(is, l);)
        if (l.rfind(prefix, 0) == 0) out.push_back(l);
    return out;
}

std::string bench_csv_path()
{
    const char *p = std::getenv("ORCA_ARRANGE_BENCH_CSV");
    return p ? p : "arrange_bench.csv";
}

void write_bench_line(const std::string &scenario, Level level, bool rot, const Run &r, const Metrics &m)
{
    static bool truncated = false;
    std::ofstream f(bench_csv_path(), truncated ? std::ios::app : std::ios::trunc);
    if (!truncated) {
        f << "scenario,geometry,rotate,items,on_bed0,pile_w_mm,pile_h_mm,seconds\n";
        truncated = true;
    }
    char line[256];
    std::snprintf(line, sizeof line, "%s,%s,%d,%zu,%d,%.1f,%.1f,%.3f", scenario.c_str(), level_name(level), int(rot),
                  r.items.size(), m.on_bed0, m.pile_w, m.pile_h, r.seconds);
    f << line << "\n";
    std::cout << "arrange-bench " << line << std::endl;
}

// Levels to benchmark: ORCA_ARRANGE_BENCH_LEVELS="fast,precise", all but Exhaustive by default.
std::vector<Level> bench_levels()
{
    const char *sel = std::getenv("ORCA_ARRANGE_BENCH_LEVELS");
    const std::string s = sel ? sel : "fast,balanced,precise";
    std::vector<Level> out;
    for (Level l : {Level::Fast, Level::Balanced, Level::Precise, Level::Exhaustive})
        if (("," + s + ",").find("," + std::string(level_name(l)) + ",") != std::string::npos)
            out.push_back(l);
    return out;
}

// Runs per scenario, ORCA_ARRANGE_BENCH_REPEAT (default 1); the fastest time is kept, as a
// laptop's run-to-run noise exceeds most single optimisations.
int bench_repeats()
{
    const char *n = std::getenv("ORCA_ARRANGE_BENCH_REPEAT");
    return n ? std::max(1, std::atoi(n)) : 1;
}

} // namespace

TEST_CASE("Every geometry level keeps items apart and on the bed", "[Arrange][ArrangeGeometry]")
{
    const std::string name   = GENERATE(as<std::string>{}, "cyl30", "L16", "C20", "mixed24", "rings");
    const Level       level  = GENERATE(Level::Fast, Level::Balanced, Level::Precise);
    const bool        rotate = GENERATE(false, true);
    DYNAMIC_SECTION(name << " " << level_name(level) << " rotate=" << rotate)
    {
        require_valid_layout(run_arrange(name, level, rotate).items);
    }
}

TEST_CASE("Precise arrange nests concave parts tighter than the convex hull does", "[Arrange][ArrangeGeometry]")
{
    const int fast = metrics(run_arrange("cap-L", Level::Fast, false).items).on_bed0;
    const Run prec = run_arrange("cap-L", Level::Precise, false);
    require_valid_layout(prec.items);
    CHECK(metrics(prec.items).on_bed0 > fast);
}

TEST_CASE("Arrange geometry levels read back from their stored names", "[Arrange][ArrangeGeometry]")
{
    const Level level = GENERATE(Level::Fast, Level::Balanced, Level::Precise, Level::Exhaustive);
    CHECK(arrange_geometry_from_name(arrange_geometry_name(level)) == level);
    CHECK(arrange_geometry_from_name("no such level") == Level::Fast);
}

TEST_CASE("Printing by object arranges by convex hulls at every level", "[Arrange][ArrangeGeometry]")
{
    ArrangeParams p;
    p.geometry     = Level::Precise;
    p.is_seq_print = true;
    CHECK(effective_geometry(p) == Level::Fast);
    p.is_seq_print = false;
    CHECK(effective_geometry(p) == Level::Precise);
}

// A Precise item can take seconds per rotation, so cancelling must not wait for all of them.
TEST_CASE("Precise arrange checks for cancel before every rotation it tries", "[Arrange][ArrangeGeometry]")
{
    ArrangePolygons items = scenario_items("C20");
    // Precise tries 8 rotations for every item after the first.
    CHECK(stop_polls(items, Level::Precise, true) >= int(items.size() - 1) * 8);
}

// A disc looks the same at every 45 degree step, so trying it at 8 angles is wasted work.
TEST_CASE("Precise arrange tries a disc at one angle only", "[Arrange][ArrangeGeometry]")
{
    ArrangePolygons layout = scenario_items("cyl30"), unrotated = scenario_items("cyl30");
    const int       with_rotation = stop_polls(layout, Level::Precise, true);
    const int       one_angle     = stop_polls(unrotated, Level::Precise, false);
    require_valid_layout(layout);
    // Precise also runs Balanced and Fast. Without rotation each tries every item at one angle.
    // With it, Fast tries 4 angles (3 more polls per item; it has no symmetry check), while
    // Precise's 8 and Balanced's 4 angles would add 10 more per item without one.
    CHECK(with_rotation <= one_angle + 4 * int(layout.size()));
}

TEST_CASE("Balanced arrange places small discs inside the holes of rings", "[Arrange][ArrangeGeometry]")
{
    CHECK(discs_in_ring_holes(run_arrange("rings", Level::Fast, false).items) == 0);
    const Run balanced = run_arrange("rings", Level::Balanced, false);
    require_valid_layout(balanced.items);
    CHECK(discs_in_ring_holes(balanced.items) > 0);
}

// Straight-sided outlines lose nothing to simplification, so they get no margin on top of
// the spacing and nest as tight as the spacing allows.
TEST_CASE("Precise arrange packs straight-sided parts at the set spacing", "[Arrange][ArrangeGeometry]")
{
    const std::string name = GENERATE(as<std::string>{}, "L16", "C20");
    DYNAMIC_SECTION(name)
    {
        const Run r = run_arrange(name, Level::Precise, false);
        require_valid_layout(r.items);
        CHECK(closest_gap_mm(r.items) <= SPACING_MM + 0.1);
    }
}

// Interlocking C shapes only pays off when it makes the whole pile smaller. Lookahead judges
// each placement by the layout it leads to, where a greedy placer hooks parts together
// wherever that scores best for one item.
TEST_CASE("Precise arrange interlocks C shapes into a smaller pile than Fast", "[Arrange][ArrangeGeometry]")
{
    const Run fast    = run_arrange("C20", Level::Fast, true);
    const Run precise = run_arrange("C20", Level::Precise, true);
    require_valid_layout(precise.items);
    CHECK(metrics(precise.items).on_bed0 >= metrics(fast.items).on_bed0);
    // An arm is 10 mm wide; 100 mm2 is a real hook, not a corner graze.
    CHECK(interlocked_items(precise.items, 100.) >= 4);
    CHECK(pile_hull_area_mm2(precise.items) < pile_hull_area_mm2(fast.items));
}

// A slower level must not undo what a faster one achieves on the same plate.
TEST_CASE("Each geometry level is never worse than the one below it", "[Arrange][ArrangeGeometry]")
{
    const std::string name   = GENERATE(as<std::string>{}, "cyl30", "cap-cyl", "L16", "cap-L", "C20", "mixed24", "cap-mixed",
                                      "rings", "rings-big");
    const bool        rotate = GENERATE(false, true);
    DYNAMIC_SECTION(name << " rotate=" << rotate)
    {
        const auto fast     = arrangement_cost(run_arrange(name, Level::Fast, rotate).items);
        const auto balanced = arrangement_cost(run_arrange(name, Level::Balanced, rotate).items);
        const auto precise  = arrangement_cost(run_arrange(name, Level::Precise, rotate).items);
        INFO("items on bed 0: fast " << -std::get<2>(fast) << ", balanced " << -std::get<2>(balanced) << ", precise "
                                     << -std::get<2>(precise));
        CHECK(balanced <= fast);
        CHECK(precise <= balanced);
    }
}

// Hidden from the default run: Exhaustive takes up to half a minute per plate, which would
// add over a minute to every ctest. Run with libslic3r_tests "[ArrangeSlow]".
TEST_CASE("Exhaustive arrange is never worse than Precise", "[Arrange][ArrangeGeometry][ArrangeSlow][.]")
{
    const std::string name   = GENERATE(as<std::string>{}, "cap-mixed", "cap-L", "C20", "mixed24");
    const bool        rotate = GENERATE(false, true);
    DYNAMIC_SECTION(name << " rotate=" << rotate)
    {
        const Run precise    = run_arrange(name, Level::Precise, rotate);
        const Run exhaustive = run_arrange(name, Level::Exhaustive, rotate);
        require_valid_layout(exhaustive.items);
        const auto p = arrangement_cost(precise.items), e = arrangement_cost(exhaustive.items);
        INFO("precise " << -std::get<2>(p) << " on bed 0, exhaustive " << -std::get<2>(e));
        CHECK(e <= p);
    }
}

// Exhaustive's other runs (scoring weights, starting point, item order) must find layouts
// Precise misses, or the extra time buys nothing. Hidden from the default run for the same
// reason as the test above.
TEST_CASE("Exhaustive arrange fits more items than Precise on crowded plates", "[Arrange][ArrangeGeometry][ArrangeSlow][.]")
{
    const auto [name, rotate] = GENERATE(table<std::string, bool>({{"cap-L", false}, {"cap-mixed", true}}));
    DYNAMIC_SECTION(name << " rotate=" << rotate)
    {
        const Run precise    = run_arrange(name, Level::Precise, rotate);
        const Run exhaustive = run_arrange(name, Level::Exhaustive, rotate);
        require_valid_layout(exhaustive.items);
        CHECK(metrics(exhaustive.items).on_bed0 > metrics(precise.items).on_bed0);
    }
}

// Fast is Orca's original arrange. Its layouts must not change. Regenerate the file with
// ORCA_WRITE_ARRANGE_GOLDEN=1 only when that behaviour is changed on purpose.
TEST_CASE("Fast arrange gives the recorded layouts", "[Arrange][Golden]")
{
    std::string all;
    for (const std::string &name : ALL_SCENARIOS)
        for (bool rot : {false, true})
            all += fast_layout_dump(name, rot, run_arrange(name, Level::Fast, rot).items);

    if (std::getenv("ORCA_WRITE_ARRANGE_GOLDEN") != nullptr) {
        std::ofstream(golden_path()) << all;
        WARN("wrote " << golden_path());
    }
    std::ifstream      in(golden_path());
    std::ostringstream golden;
    golden << in.rdbuf();
    REQUIRE_FALSE(golden.str().empty());
    for (const std::string &name : ALL_SCENARIOS) {
        INFO(name);
        CHECK(lines_of(all, name + " ") == lines_of(golden.str(), name + " "));
    }
}

// Benchmark, hidden from the default run. Run it with libslic3r_tests "[ArrangeBench]";
// ORCA_ARRANGE_BENCH="rings,C20" narrows the plates.
TEST_CASE("Arrange benchmark", "[Arrange][ArrangeBench][.]")
{
    std::vector<std::string> names = ALL_SCENARIOS;
    if (const char *sel = std::getenv("ORCA_ARRANGE_BENCH")) {
        names.clear();
        std::istringstream is(sel);
        for (std::string n; std::getline(is, n, ',');) names.push_back(n);
    }
    for (const std::string &name : names)
        for (Level level : bench_levels())
            for (bool rot : {false, true}) {
                Run r = run_arrange(name, level, rot);
                for (int i = 1; i < bench_repeats(); ++i)
                    r.seconds = std::min(r.seconds, run_arrange(name, level, rot).seconds);
                write_bench_line(name, level, rot, r, metrics(r.items));
                INFO(name << " " << level_name(level) << " rotate=" << rot);
                require_valid_layout(r.items);
            }
}
